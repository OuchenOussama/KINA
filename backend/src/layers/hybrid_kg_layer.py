import os
import gc
import re
import json
import pickle
import logging
import numpy as np
import pandas as pd
import networkx as nx
import faiss
from typing import List, Tuple, Dict
from sentence_transformers import SentenceTransformer
import torch
import nltk
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
from nltk.corpus import stopwords
from collections import defaultdict

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class HybridPharmaceuticalKG:
    def __init__(self,
                 csv_path: str,
                 embedding_model: str = 'all-MiniLM-L6-v2',
                 embedding_dim: int = 64,
                 walk_length: int = 10,
                 num_walks: int = 5,
                 max_nodes_per_type: int = 20000,
                 batch_size: int = 64,
                 text_weight: float = 0.6,
                 structure_weight: float = 0.6,
                 checkpoint_dir: str = 'embeddings',
                 enable_text_embeddings: bool = True,
                 enable_node_embeddings: bool = True,
                 use_precomputed: bool = True):
        """
        Initialize the Hybrid Pharmaceutical Knowledge Graph.
        
        Args:
            csv_path: Path to the input CSV file
            embedding_model: SentenceTransformer model for text embeddings
            embedding_dim: Dimension for structural embeddings
            walk_length: Length of random walks for node2vec
            num_walks: Number of walks per node
            max_nodes_per_type: Maximum nodes per type to limit memory usage
            batch_size: Batch size for text embedding generation
            text_weight: Weight for text-based similarity
            structure_weight: Weight for structural similarity
            checkpoint_dir: Directory to store/load embeddings
            enable_text_embeddings: Whether to use text embeddings
            enable_node_embeddings: Whether to use structural embeddings
            use_precomputed: Whether to use cached precomputed data
        """
        self.csv_path = csv_path
        self.embedding_model = embedding_model
        self.embedding_dim = embedding_dim
        self.walk_length = walk_length
        self.num_walks = num_walks
        self.max_nodes_per_type = max_nodes_per_type
        self.batch_size = batch_size
        self.text_weight = text_weight
        self.structure_weight = structure_weight
        self.checkpoint_dir = checkpoint_dir
        self.enable_text_embeddings = enable_text_embeddings
        self.enable_node_embeddings = enable_node_embeddings
        self.use_precomputed = use_precomputed

        # Initialize components
        self.df = None
        self.G = None
        self.node_embeddings = None
        self.text_embeddings = None
        self.node_texts = {}
        self.node_mapping = {}
        self.reverse_mapping = {}
        self.index_structural = None
        self.index_text = None

        # NLTK resources
        try:
            nltk.data.find('tokenizers/punkt')
            nltk.data.find('corpora/stopwords')
        except LookupError:
            logger.info("Downloading NLTK resources...")
            try:
                nltk.download('punkt', quiet=True)
                nltk.download('stopwords', quiet=True)
                nltk.download('punkt_tab', quiet=True)
            except Exception as e:
                logger.error(f"Failed to download NLTK resources: {e}")

        self.stemmer = PorterStemmer()
        self.stop_words = set(stopwords.words('english'))

        # Initialize text model
        try:
            self.text_model = SentenceTransformer(
                embedding_model,
                device='cuda' if torch.cuda.is_available() else 'cpu'
            )
        except Exception as e:
            logger.error(f"Failed to initialize SentenceTransformer: {e}")
            raise RuntimeError("Cannot initialize text model")

        try:
            os.makedirs(checkpoint_dir, exist_ok=True)
        except Exception as e:
            logger.error(f"Failed to create checkpoint directory {checkpoint_dir}: {e}")
            raise RuntimeError(f"Cannot create directory {checkpoint_dir}")

        logger.info("Hybrid Pharmaceutical Knowledge Graph initialized.")

    def preprocess_data(self, verbose: bool = True, chunksize: int = 500) -> pd.DataFrame:
        """Preprocess CSV data with chunking and caching."""
        cache_path = os.path.join(self.checkpoint_dir, 'preprocessed_data.pkl')
        
        if self.use_precomputed:
            try:
                if os.path.exists(cache_path):
                    self.df = pd.read_pickle(cache_path)
                    if verbose:
                        logger.info(f"Loaded preprocessed data from {cache_path}")
                    return self.df
            except Exception as e:
                logger.warning(f"Failed to load preprocessed data from {cache_path}: {e}")

        if verbose:
            logger.info(f"Preprocessing data from {self.csv_path}...")

        try:
            chunks = []
            categorical_columns = ['BrandName', 'Composition', 'TherapeuticClass', 'Dosage']
            text_columns = ['Uses', 'Packaging', 'Considerations', 'Contraindications', 'Form']

            for chunk in pd.read_csv(self.csv_path, chunksize=chunksize, encoding='utf-8', on_bad_lines='skip'):
                for col in text_columns:
                    if col in chunk.columns:
                        chunk[col] = chunk[col].astype(str).replace(['nan', 'None'], np.nan)
                        chunk[col] = chunk[col].apply(
                            lambda x: re.sub(r'\s+', ' ', x.strip()) if pd.notna(x) else x
                        )
                for col in categorical_columns:
                    if col in chunk.columns:
                        unique_vals = chunk[col].dropna().unique().tolist()
                        if 'unknown' not in unique_vals:
                            unique_vals.append('unknown')
                        chunk[col] = pd.Categorical(chunk[col], categories=unique_vals)
                chunk = chunk.fillna('unknown')
                chunks.append(chunk)
                gc.collect()

            self.df = pd.concat(chunks, ignore_index=True)
            try:
                self.df.to_pickle(cache_path)
                if verbose:
                    logger.info(f"Saved preprocessed data to {cache_path}")
            except Exception as e:
                logger.warning(f"Failed to save preprocessed data to {cache_path}: {e}")
            return self.df
        except Exception as e:
            logger.error(f"Failed to preprocess data from {self.csv_path}: {e}")
            return None

    def _extract_node_text(self, node_name: str) -> str:
        """Extract text representation for nodes."""
        try:
            if node_name in self.node_texts:
                return self.node_texts[node_name]

            if ':' not in node_name:
                return f"pharmaceutical drug {node_name}"

            node_type, node_value = node_name.split(':', 1)
            node_type_templates = {
                'TherapeuticClass': [f"under therapeutic class of {node_value}"],
                'Contraindications': [f"contraindicated in {node_value}"],
                'Form': [f"form of {node_value}"],
                'Packaging': [f"packaged as {node_value}"],
                'Considerations': [f"requires consideration {node_value}"],
                'Uses': [f"treatment of {node_value}"],
                'Composition': [f"contains active ingredient {node_value}"],
                'Dosage': [f"dosage strength of {node_value}"],
                # 'Price': f"sold for {node_value}"
            }
            
            templates = node_type_templates.get(node_type, [f"pharmaceutical attribute {node_value}"])
            if node_type in ['Uses', 'Considerations', 'Contraindications']:
                actual_text = self._get_actual_node_content(node_name)
                if actual_text and len(actual_text) > 10:
                    return f"{templates[0]}. {actual_text}"  # ← Use templates[0]
            return templates[0] 
        
        except Exception as e:
            logger.error(f"Error extracting node text for {node_name}: {e}")
            return ""

    def _get_actual_node_content(self, node_name: str) -> str:
        """Retrieve actual text content for text-based nodes."""
        try:
            if ':' not in node_name:
                return ""

            node_type, node_id = node_name.split(':', 1)
            if node_id.isdigit() and self.df is not None:
                idx = int(node_id)
                if idx < len(self.df) and node_type in self.df.columns:
                    content = self.df.loc[idx, node_type]
                    if pd.notna(content) and content != 'unknown':
                        return re.sub(r'\s+', ' ', str(content).strip())[:200]
            return ""
        except Exception as e:
            logger.error(f"Error getting node content for {node_name}: {e}")
            return ""

    def build_graph(self, verbose: bool = True) -> nx.Graph:
        """Build knowledge graph with caching."""
        if self.df is None:
            self.preprocess_data(verbose)

        cache_path = os.path.join(self.checkpoint_dir, 'knowledge_graph.pkl')
        
        if self.use_precomputed:
            try:
                if os.path.exists(cache_path):
                    with open(cache_path, 'rb') as f:
                        cached_data = pickle.load(f)
                        self.G = cached_data['graph']
                        self.node_texts = cached_data['node_texts']
                    if verbose:
                        logger.info(f"Loaded graph from {cache_path}")
                    return self.G
            except Exception as e:
                logger.warning(f"Failed to load graph from {cache_path}: {e}")

        if verbose:
            logger.info("Building knowledge graph...")

        try:
            self.G = nx.Graph()
            unique_nodes = set()
            self.node_texts = {}

            df_sample = self.df.head(self.max_nodes_per_type * 10)
            brand_names = df_sample['BrandName'].unique()[:self.max_nodes_per_type]
            unique_nodes.update(brand_names)

            single_value_columns = ['TherapeuticClass', 'Dosage', 'Composition']
            for col in single_value_columns:
                if col in df_sample.columns:
                    unique_vals = df_sample[col].iloc[:self.max_nodes_per_type]
                    unique_nodes.update(f"{col}:{val}" for val in unique_vals if val != 'unknown')

            text_columns = ['Uses', 'Considerations', 'Contraindications', 'Packaging', 'Form']
            for col in text_columns:
                if col not in df_sample.columns:
                    continue
                valid_rows = df_sample[
                    (df_sample[col] != 'unknown') & (df_sample[col].str.len() > 5)
                ].iloc[:self.max_nodes_per_type].copy()
                for idx in valid_rows.index:
                    node_name = f"{col}:{idx}"
                    unique_nodes.add(node_name)
                    raw_text = valid_rows.loc[idx, col]
                    self.node_texts[node_name] = re.sub(r'\s+', ' ', raw_text.strip())

            for node in unique_nodes:
                node_type = 'BrandName' if ':' not in node else node.split(':', 1)[0]
                self.G.add_node(node, type=node_type)

            edges = []
            chunk_size = 1000
            for start in range(0, min(len(df_sample), self.max_nodes_per_type * 10), chunk_size):
                chunk = df_sample[start:start + chunk_size]
                for idx, row in chunk.iterrows():
                    brand_name = row['BrandName']
                    if brand_name == 'unknown':
                        continue
                    for col in df_sample.columns:
                        if col == 'BrandName':
                            continue
                        elif col in ['Uses', 'Considerations', 'Contraindications', 'Packaging', 'Form']:
                            if row[col] != 'unknown':
                                node_id = f"{col}:{idx}"
                                if node_id in unique_nodes:
                                    edges.append((brand_name, node_id, col))
                        else:
                            val = row[col]
                            if val != 'unknown':
                                attr_node = f"{col}:{val}"
                                if attr_node in unique_nodes:
                                    edges.append((brand_name, attr_node, col))
                gc.collect()

            self.G.add_edges_from((u, v, {'type': t, 'weight': 1.0}) for u, v, t in edges)

            try:
                cache_data = {'graph': self.G, 'node_texts': self.node_texts}
                with open(cache_path, 'wb') as f:
                    pickle.dump(cache_data, f, pickle.HIGHEST_PROTOCOL)
                if verbose:
                    logger.info(f"Saved graph to {cache_path}")
            except Exception as e:
                logger.warning(f"Failed to save graph to {cache_path}: {e}")

            if verbose:
                logger.info(f"Graph built with {self.G.number_of_nodes()} nodes and {self.G.number_of_edges()} edges")
            return self.G
        except Exception as e:
            logger.error(f"Failed to build graph: {e}")
            return None

    def generate_structural_embeddings(self, verbose: bool = True) -> Dict:
        """Generate structural embeddings using Node2Vec."""
        if not self.enable_node_embeddings:
            self.node_embeddings = {node: np.zeros(self.embedding_dim) for node in self.G.nodes()}
            if verbose:
                logger.info("Node embeddings disabled. Using zero embeddings.")
            return self.node_embeddings

        cache_path = os.path.join(self.checkpoint_dir, 'structural_embeddings.pkl')
        
        if self.use_precomputed:
            try:
                if os.path.exists(cache_path):
                    with open(cache_path, 'rb') as f:
                        self.node_embeddings = pickle.load(f)
                    if verbose:
                        logger.info(f"Loaded structural embeddings from {cache_path}")
                    return self.node_embeddings
            except Exception as e:
                logger.warning(f"Failed to load structural embeddings from {cache_path}: {e}")

        if verbose:
            logger.info("Generating structural embeddings...")

        try:
            from node2vec import Node2Vec
            node2vec = Node2Vec(
                self.G,
                dimensions=self.embedding_dim,
                walk_length=self.walk_length,
                num_walks=self.num_walks,
                p=1.0,
                q=1.0,
                workers=2
            )
            model = node2vec.fit(window=5, min_count=1, batch_words=4, epochs=8)
            self.node_embeddings = {node: model.wv[node] for node in self.G.nodes()}
            del model, node2vec
        except Exception as e:
            logger.warning(f"Node2Vec failed: {e}. Using random embeddings.")
            self.node_embeddings = {
                node: np.random.normal(0, 0.1, self.embedding_dim) for node in self.G.nodes()
            }

        try:
            with open(cache_path, 'wb') as f:
                pickle.dump(self.node_embeddings, f)
            if verbose:
                logger.info(f"Saved structural embeddings to {cache_path}")
        except Exception as e:
            logger.warning(f"Failed to save structural embeddings to {cache_path}: {e}")
        
        gc.collect()
        if verbose:
            logger.info(f"Generated structural embeddings for {len(self.node_embeddings)} nodes")
        return self.node_embeddings

    def generate_text_embeddings(self, verbose: bool = True) -> Dict:
        """Generate text embeddings using SentenceTransformer."""
        cache_path = os.path.join(self.checkpoint_dir, 'textual_embeddings.pkl')
        
        if self.use_precomputed:
            try:
                if os.path.exists(cache_path):
                    with open(cache_path, 'rb') as f:
                        self.text_embeddings = pickle.load(f)
                    if verbose:
                        logger.info(f"Loaded text embeddings from {cache_path}")
                    return self.text_embeddings
            except Exception as e:
                logger.warning(f"Failed to load text embeddings from {cache_path}: {e}")

        if not self.enable_text_embeddings:
            self.text_embeddings = {node: np.zeros(384) for node in self.G.nodes()}
            if verbose:
                logger.info("Text embeddings disabled. Using zero embeddings.")
            return self.text_embeddings

        if verbose:
            logger.info("Generating text embeddings...")

        try:
            texts = []
            nodes = []
            for node in self.G.nodes():
                node_text = self._extract_node_text(node)
                texts.append(node_text)
                nodes.append(node)

            self.text_embeddings = {}
            for i in range(0, len(texts), self.batch_size):
                batch_texts = texts[i:i + self.batch_size]
                batch_nodes = nodes[i:i + self.batch_size]
                try:
                    embeddings = self.text_model.encode(
                        batch_texts,
                        convert_to_numpy=True,
                        batch_size=self.batch_size,
                        show_progress_bar=False
                    )
                    for node, embedding in zip(batch_nodes, embeddings):
                        self.text_embeddings[node] = embedding
                except Exception as e:
                    logger.warning(f"Error in text embedding batch {i//self.batch_size}: {e}")
                    for node in batch_nodes:
                        self.text_embeddings[node] = np.zeros(384)
                gc.collect()

            try:
                with open(cache_path, 'wb') as f:
                    pickle.dump(self.text_embeddings, f)
                if verbose:
                    logger.info(f"Saved text embeddings to {cache_path}")
            except Exception as e:
                logger.warning(f"Failed to save text embeddings to {cache_path}: {e}")
        except Exception as e:
            logger.error(f"Failed to generate text embeddings: {e}")
            self.text_embeddings = {node: np.zeros(384) for node in self.G.nodes()}

        if verbose:
            logger.info(f"Generated text embeddings for {len(self.text_embeddings)} nodes")
        return self.text_embeddings

    def build_faiss_indices(self, verbose: bool = True):
        """Build FAISS indices for efficient search."""
        if not self.node_embeddings or not self.text_embeddings:
            logger.error("No embeddings found. Please generate embeddings first.")
            return

        if verbose:
            logger.info("Building FAISS indices...")

        try:
            self.node_mapping = {node: i for i, node in enumerate(self.G.nodes())}
            self.reverse_mapping = {i: node for node, i in self.node_mapping.items()}

            if self.enable_node_embeddings:
                structural_matrix = np.array([
                    self.node_embeddings[node] for node in self.node_mapping.keys()
                ]).astype('float32')
                faiss.normalize_L2(structural_matrix)
                self.index_structural = faiss.IndexFlatL2(structural_matrix.shape[1])
                self.index_structural.add(structural_matrix)
                del structural_matrix
            else:
                self.index_structural = None

            if self.enable_text_embeddings:
                text_matrix = np.array([
                    self.text_embeddings[node] for node in self.node_mapping.keys()
                ]).astype('float32')
                faiss.normalize_L2(text_matrix)
                self.index_text = faiss.IndexFlatL2(text_matrix.shape[1])
                self.index_text.add(text_matrix)
                del text_matrix
            else:
                self.index_text = None

            if verbose:
                logger.info(f"Built FAISS indices for {len(self.node_mapping)} nodes")
        except Exception as e:
            logger.error(f"Failed to build FAISS indices: {e}")
            self.index_structural = None
            self.index_text = None

        gc.collect()

    def load_model(self, verbose: bool = True):
        """Load precomputed model components from embeddings folder."""
        try:
            if self.preprocess_data(verbose) is None:
                raise RuntimeError("Failed to preprocess data")
            if self.build_graph(verbose) is None:
                raise RuntimeError("Failed to build graph")
            if self.generate_structural_embeddings(verbose) is None:
                raise RuntimeError("Failed to generate structural embeddings")
            if self.generate_text_embeddings(verbose) is None:
                raise RuntimeError("Failed to generate text embeddings")
            self.build_faiss_indices(verbose)
            if verbose:
                logger.info("Model loaded successfully")
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            raise

    def find_drugs_smart(self, query: str, k: int = 10) -> List[Tuple[str, float, List[str], Dict[str, List[str]]]]:
        """
        Find drugs matching the query using hybrid search with immediate node information.
        
        Args:
            query: User query string
            k: Number of results to return
            
        Returns:
            List of tuples (drug_name, score, reasons, immediate_nodes_info)
            where immediate_nodes_info is a dict with keys like 'Composition', 'TherapeuticClass', etc.
        """
        if not self.enable_node_embeddings and not self.enable_text_embeddings:
            logger.error("Both embeddings disabled. Cannot perform search.")
            return []

        if (self.enable_node_embeddings and not self.index_structural) or \
        (self.enable_text_embeddings and not self.index_text):
            logger.error("Required indices not built. Please load model first.")
            return []

        try:
            drug_results = defaultdict(lambda: [0.0, []])

            # Structural search
            if self.enable_node_embeddings and self.index_structural:
                query_nodes = [
                    node for node in self.G.nodes()
                    if query.lower() in self._extract_node_text(node).lower()
                ]
                for query_node in query_nodes[:10]:
                    if query_node in self.node_mapping:
                        try:
                            query_embedding = self.node_embeddings[query_node].reshape(1, -1).astype('float32')
                            faiss.normalize_L2(query_embedding)
                            distances, indices = self.index_structural.search(query_embedding, k)
                            for idx, dist in zip(indices[0], distances[0]):
                                node_name = self.reverse_mapping[idx]
                                node_type = self.G.nodes[node_name].get('type', 'Unknown')
                                if node_type == 'BrandName':
                                    drug_results[node_name][0] += (1 - dist) * self.structure_weight
                                    drug_results[node_name][1].append(f"Structural match via {query_node}")
                                else:
                                    for neighbor in self.G.neighbors(node_name):
                                        if self.G.nodes[neighbor].get('type') == 'BrandName':
                                            drug_results[neighbor][0] += (1 - dist) * self.structure_weight * 0.8
                                            drug_results[neighbor][1].append(f"Connected via {node_name}")
                        except Exception as e:
                            logger.warning(f"Error in structural search for node {query_node}: {e}")

            # Text search
            if self.enable_text_embeddings and self.index_text:
                try:
                    query_embedding = self.text_model.encode([query], convert_to_numpy=True).astype('float32')
                    faiss.normalize_L2(query_embedding)
                    distances, indices = self.index_text.search(query_embedding, k)
                    for idx, dist in zip(indices[0], distances[0]):
                        node_name = self.reverse_mapping[idx]
                        node_type = self.G.nodes[node_name].get('type', 'Unknown')
                        if node_type == 'BrandName':
                            drug_results[node_name][0] += (1 - dist) * self.text_weight
                            drug_results[node_name][1].append("Text similarity match")
                        else:
                            for neighbor in self.G.neighbors(node_name):
                                if self.G.nodes[neighbor].get('type') == 'BrandName':
                                    drug_results[neighbor][0] += (1 - dist) * self.text_weight * 0.8
                                    drug_results[neighbor][1].append(f"Text match via {node_name}")
                except Exception as e:
                    logger.warning(f"Error in text search: {e}")

            # Get immediate nodes information for each drug
            final_results = []
            for drug, (score, reasons) in drug_results.items():
                immediate_nodes_info = self._get_immediate_nodes_info(drug)
                final_results.append((drug, score, reasons, immediate_nodes_info))
            
            final_results.sort(key=lambda x: x[1], reverse=True)
            
            gc.collect()
            return final_results[:k]
        except Exception as e:
            logger.error(f"Error in find_drugs_smart: {e}")
            return []

    def _get_immediate_nodes_info(self, drug_name: str) -> Dict[str, List[str]]:
        """
        Get immediate neighboring nodes information for a drug.
        
        Args:
            drug_name: Name of the drug (BrandName node)
            
        Returns:
            Dictionary with node types as keys and lists of values as values
        """
        immediate_info = {
            'Composition': [],
            'TherapeuticClass': [],
            'Price': [],
            'Dosage': [],
            'Uses': [],
            'Contraindications': [],
            'Considerations': [],
            'Form': [],
            'Packaging': []
        }
        
        try:
            if drug_name not in self.G:
                logger.warning(f"Drug {drug_name} not found in graph")
                return immediate_info
                
            # Get all neighbors of the drug
            neighbors = list(self.G.neighbors(drug_name))
            
            for neighbor in neighbors:
                # Extract node type and value
                if ':' in neighbor:
                    node_type, node_value = neighbor.split(':', 1)
                    
                    if node_type in immediate_info:
                        # For text-based nodes (Uses, Considerations, etc.), get actual content
                        if node_type in ['Uses', 'Considerations', 'Contraindications', 'Packaging', 'Form']:
                            actual_content = self._get_actual_node_content(neighbor)
                            if actual_content:
                                immediate_info[node_type].append(actual_content)
                            else:
                                # Fallback to node value if actual content not available
                                immediate_info[node_type].append(node_value)
                        else:
                            # For categorical nodes, use the value directly
                            immediate_info[node_type].append(node_value)
            
            # Remove duplicates while preserving order
            for key in immediate_info:
                immediate_info[key] = list(dict.fromkeys(immediate_info[key]))
                
        except Exception as e:
            logger.error(f"Error getting immediate nodes info for {drug_name}: {e}")
        
        return immediate_info

    def clear_memory(self):
        """Clear memory-intensive components but keep FAISS indices."""
        try:
            self.node_embeddings = None
            self.text_embeddings = None
            self.df = None
            self.node_texts = {}
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception as e:
            logger.error(f"Error clearing memory: {e}")