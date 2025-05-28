import networkx as nx
import pandas as pd
import numpy as np
import re
import os
import json
import pickle
import logging
import nltk
from collections import defaultdict
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
from nltk.corpus import stopwords
import faiss
from sentence_transformers import SentenceTransformer
import torch
import itertools

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class HybridPharmaceuticalKG:
    def __init__(self,
                 csv_path: str,
                 embedding_dim: int = 64,
                 walk_length: int = 50,
                 num_walks: int = 5,
                 p: float = 1.0,
                 q: float = 1.0,
                 text_model: str = 'all-MiniLM-L6-v2',
                 synonym_weight: float = 0.9,
                 text_weight: float = 0.3,
                 structure_weight: float = 0.7,
                 batch_size: int = 16,
                 max_text_length: int = 512,
                 max_nodes_per_type: int = 20000,
                 enable_text_embeddings: bool = True):
                                                        
        self.csv_path = csv_path
        self.embedding_dim = embedding_dim
        self.walk_length = walk_length
        self.num_walks = num_walks
        self.p = p
        self.q = q
        self.synonym_weight = synonym_weight
        self.text_weight = text_weight
        self.structure_weight = structure_weight
        self.batch_size = batch_size
        self.max_text_length = max_text_length
        self.max_nodes_per_type = max_nodes_per_type
        self.enable_text_embeddings = enable_text_embeddings

        self.df = None
        self.G = nx.Graph()
        self.node_embeddings = None
        self.text_embeddings = None
        self.node_texts = {}
        self.synonyms = {}
        self.index_structural = None
        self.index_text = None
        self.node_mapping = {}
        self.reverse_mapping = {}

        self.text_model = SentenceTransformer(text_model, device='cuda' if torch.cuda.is_available() else 'cpu')
        try:
            nltk.data.find('corpora/stopwords')
            nltk.data.find('tokenizers/punkt')
            nltk.download('punkt_tab')
        except LookupError:
            logger.info("Downloading NLTK resources...")
            nltk.download('stopwords', quiet=True)
            nltk.download('punkt', quiet=True)

        self.stemmer = PorterStemmer()
        self.stop_words = set(stopwords.words('english'))   

        self.medical_synonyms = {
            'acne': ['pimples', 'zits', 'skin breakout', 'comedones', 'acne vulgaris'],
            'allergy': ['allergic reaction', 'hypersensitivity', 'hay fever', 'atopy', 'anaphylaxis', 'contact dermatitis', 'urticaria'],
            'anemia': ['low hemoglobin', 'iron deficiency', 'sideropenia', 'hemolytic anemia', 'pernicious anemia', 'thalassemia', 'sickle cell anemia'],
            'antibiotic': ['antibacterial drug', 'antimicrobial', 'bactericidal', 'bacteriostatic', 'penicillin', 'cephalosporin', 'tetracycline'],
            'antidepressant': ['mood stabilizer', 'SSRI', 'SNRI', 'tricyclic antidepressant', 'MAOI'],
            'antifungal': ['fungus treatment', 'antimycotic', 'azole', 'polyene', 'echinocandin'],
            'antihistamine': ['allergy medicine', 'histamine blocker', 'H1 antagonist', 'H2 antagonist', 'cetirizine', 'loratadine'],
            'antipyretic': ['fever reducer', 'NSAID', 'acetaminophen', 'paracetamol', 'salicylate'],
            'tuberculosis': ['TB', 'mycobacterium tuberculosis', 'pulmonary TB', 'latent TB'],
            'ulcerative colitis': ['inflammatory bowel disease', 'colitis', 'bowel inflammation', 'proctitis']
        }
        self._build_synonym_lookup()
        logger.info("Hybrid Knowledge Graph System initialized.")

    def _build_synonym_lookup(self):
        self.synonym_lookup = defaultdict(set)
        for main_term, synonyms in self.medical_synonyms.items():
            self.synonym_lookup[main_term].update(synonyms)
            for synonym in synonyms:
                self.synonym_lookup[synonym].add(main_term)
                self.synonym_lookup[synonym].update(synonyms)
                self.synonym_lookup[synonym].discard(synonym)

    def _preprocess_text(self, text: str) -> str:
        if not isinstance(text, str) or text in ['nan', 'None', 'unknown']:
            return ""
        text = text.lower().translate(str.maketrans('', '', string.punctuation))
        tokens = word_tokenize(text)
        tokens = [self.stemmer.stem(token) for token in tokens if token not in self.stop_words]
        return ' '.join(tokens)

    def preprocess_data(self, verbose: bool = True, chunksize: int = 500):
        cache_path = 'preprocessed_data.pkl'
        if os.path.exists(cache_path):
            self.df = pd.read_pickle(cache_path)
            if verbose:
                logger.info(f"Loaded preprocessed data from {cache_path}")
            return self.df

        if verbose:
            logger.info(f"Loading data from {self.csv_path}...")
        chunks = []
        for chunk in pd.read_csv(self.csv_path, chunksize=chunksize, encoding='utf-8', on_bad_lines='skip'):
            for col in ['BrandName', 'Form', 'Packaging', 'TherapeuticClass']:
                if col in chunk.columns:
                    chunk[col] = chunk[col].astype('category')
            text_columns = ['BrandName', 'Form', 'Uses', 'Packaging', 'Composition', 'TherapeuticClass', 'Considerations', 'Contraindications']
            for col in text_columns:
                if col in chunk.columns:
                    chunk[col] = chunk[col].astype(str).replace(['nan', 'None'], np.nan)
                    if col in ['Uses', 'Considerations', 'Contraindications']:
                        chunk[col] = chunk[col].apply(
                            lambda x: x[:1000] if pd.notna(x) and len(str(x)) > 1000 else x
                        )
                    chunk[col] = chunk[col].apply(
                        lambda x: re.sub(r'\s+', ' ', x.strip()) if pd.notna(x) else x
                    )
            chunks.append(chunk.fillna("unknown"))
        self.df = pd.concat(chunks, ignore_index=True)
        self.df.to_pickle(cache_path)
        if verbose:
            logger.info(f"Saved preprocessed data to {cache_path}")
        return self.df

    def truncate_text(self, text, max_length=None):
        if max_length is None:
            max_length = self.max_text_length
        if not isinstance(text, str) or len(text.strip()) == 0:
            return ""
        text = re.sub(r'\s+', ' ', text.strip())
        if len(text) > max_length * 5:
            text = text[:max_length * 5]
        tokens = word_tokenize(text)[:max_length]
        return ' '.join(tokens)

    def build_graph(self, verbose: bool = True) -> nx.Graph:
        if self.df is None:
            self.preprocess_data(verbose)
        if verbose:
            logger.info("Building knowledge graph...")
        self.G = nx.Graph()
        unique_nodes = set()
        self.node_texts = {}

        unique_nodes.update(self.df['BrandName'])
        single_value_columns = ['Form', 'Packaging', 'TherapeuticClass', 'Contraindications']
        for col in single_value_columns:
            if col in self.df.columns:
                unique_nodes.update(f"{col}:{val}" for val in self.df[col] if val != 'unknown')

        text_columns = ['Uses', 'Considerations']
        for col in text_columns:
            if col not in self.df.columns:
                continue
            valid_rows = self.df[(self.df[col] != 'unknown') & (self.df[col].str.len() > 10)].copy()
            if len(valid_rows) > self.max_nodes_per_type:
                valid_rows = valid_rows.sample(n=self.max_nodes_per_type, random_state=42)
            for idx in valid_rows.index:
                node_name = f"{col}:{idx}"
                unique_nodes.add(node_name)
                self.node_texts[node_name] = self.truncate_text(valid_rows.loc[idx, col])

        if 'Composition' in self.df.columns:
            values = set(itertools.chain.from_iterable(
                re.split(r'[,;â�¢]|\band\b', str(val)) for val in self.df['Composition'] if val != 'unknown'
            ))
            unique_nodes.update(f"Composition:{val.strip()}" for val in values if val.strip())

        for node in unique_nodes:
            node_type = 'BrandName' if ':' not in node else node.split(':', 1)[0]
            self.G.add_node(node, type=node_type)

        edges = []
        sampled_text_indices = set()
        for node in unique_nodes:
            if node.startswith(('Uses:', 'Considerations:')):
                idx = int(node.split(':', 1)[1])
                sampled_text_indices.add(idx)

        for idx, row in self.df.iterrows():
            brand_name = row['BrandName']
            if brand_name == 'unknown':
                continue
            for col in self.df.columns:
                if col == 'BrandName':
                    continue
                elif col in ['Uses', 'Considerations']:
                    if idx in sampled_text_indices and row[col] != 'unknown':
                        node_id = f"{col}:{idx}"
                        if node_id in unique_nodes:
                            edges.append((brand_name, node_id, col))
                elif col == 'Composition':
                    values = re.split(r'[,;â�¢]|\band\b', str(row[col]))
                    edges.extend((brand_name, f"Composition:{val.strip()}", col) for val in values if val.strip() and val != 'unknown')
                else:
                    val = row[col]
                    if val != 'unknown':
                        edges.append((brand_name, f"{col}:{val}", col))

        self.G.add_edges_from((u, v, {'type': t}) for u, v, t in edges)
        self._build_synonym_graph()
        if verbose:
            logger.info(f"Graph built with {self.G.number_of_nodes()} nodes and {self.G.number_of_edges()} edges")
        return self.G

    def _build_synonym_graph(self):
        logger.info("Adding synonym links to graph...")
        synonym_count = 0
        nodes = list(self.G.nodes())
        for main_term, synonyms in self.medical_synonyms.items():
            for synonym in synonyms:
                matching_nodes = [
                    node for node in nodes
                    if (node.startswith('Uses:') or node.startswith('Considerations:'))
                    and (main_term.lower() in self.node_texts.get(node, '').lower()
                         or synonym.lower() in self.node_texts.get(node, '').lower())
                ]
                for node1, node2 in itertools.combinations(matching_nodes, 2):
                    self.G.add_edge(node1, node2, type='synonym', weight=self.synonym_weight)
                    synonym_count += 1
                    if node1 not in self.synonyms:
                        self.synonyms[node1] = []
                    self.synonyms[node1].append(node2)
                    if node2 not in self.synonyms:
                        self.synonyms[node2] = []
                    self.synonyms[node2].append(node1)
        logger.info(f"Added {synonym_count} synonym links")

    def generate_embeddings(self, verbose: bool = True) -> tuple[dict, dict]:
        from node2vec import Node2Vec
        if not self.G or self.G.number_of_nodes() == 0:
            logger.error("Graph is empty. Please build the graph first.")
            return {}, {}

        if verbose:
            logger.info("Generating structural embeddings with node2vec...")
        node2vec = Node2Vec(
            self.G,
            dimensions=self.embedding_dim,
            walk_length=self.walk_length,
            num_walks=self.num_walks,
            p=self.p,
            q=self.q,
            workers=1
        )
        model = node2vec.fit(window=5, min_count=1, batch_words=2, epochs=5)
        self.node_embeddings = {node: model.wv[node] for node in self.G.nodes()}
        del model, node2vec
        torch.cuda.empty_cache() if torch.cuda.is_available() else None

        if self.enable_text_embeddings:
            if verbose:
                logger.info("Generating text embeddings...")
            texts, nodes = [], []
            for node in self.G.nodes():
                text = self.node_texts.get(node, "")
                if text and len(text.strip()) > 0:
                    texts.append(text)
                    nodes.append(node)

            self.text_embeddings = {}
            for i in range(0, len(texts), self.batch_size):
                batch_texts = texts[i:i + self.batch_size]
                batch_nodes = nodes[i:i + self.batch_size]
                embeddings = self.text_model.encode(
                    batch_texts,
                    convert_to_numpy=True,
                    batch_size=self.batch_size,
                    show_progress_bar=False
                )
                for node, embedding in zip(batch_nodes, embeddings):
                    self.text_embeddings[node] = embedding

            text_dim = len(next(iter(self.text_embeddings.values()), np.zeros(384)))
            for node in self.G.nodes():
                if node not in self.text_embeddings:
                    self.text_embeddings[node] = np.zeros(text_dim)
        else:
            self.text_embeddings = {node: np.zeros(384) for node in self.G.nodes()}

        self.node_mapping = {node: i for i, node in enumerate(self.G.nodes())}
        self.reverse_mapping = {i: node for node, i in self.node_mapping.items()}
        if verbose:
            logger.info(f"Generated embeddings for {len(self.node_embeddings)} nodes")
        return self.node_embeddings, self.text_embeddings

    def build_faiss_indices(self, verbose: bool = True):
        if not self.node_embeddings or not self.text_embeddings:
            logger.error("No embeddings found. Please generate embeddings first.")
            return

        if verbose:
            logger.info("Building FAISS indices...")
        structural_matrix = np.array([
            self.node_embeddings[node] for node in self.node_mapping.keys()
        ]).astype('float32')
        faiss.normalize_L2(structural_matrix)
        self.index_structural = faiss.IndexFlatL2(structural_matrix.shape[1])
        self.index_structural.add(structural_matrix)
        del structural_matrix

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
            logger.info(f"Built indices with {self.index_structural.ntotal} vectors")

    def _semantic_search(self, query_text: str, k: int = 10) -> list[tuple[str, float]]:
        if not self.enable_text_embeddings or not self.index_text:
            query_words = set(self._preprocess_text(query_text).split())
            results = []
            for node in self.G.nodes():
                node_text = self.node_texts.get(node, "").lower()
                if node_text:
                    node_words = set(node_text.split())
                    overlap = len(query_words & node_words)
                    if overlap > 0:
                        score = overlap / len(query_words)
                        results.append((node, score))
            results.sort(key=lambda x: x[1], reverse=True)
            return results[:k]

        query_embedding = self.text_model.encode([query_text], convert_to_numpy=True, batch_size=1).astype('float32')
        faiss.normalize_L2(query_embedding)
        distances, indices = self.index_text.search(query_embedding, k)
        return [(self.reverse_mapping[idx], 1 - float(dist)) for idx, dist in zip(indices[0], distances[0])]

    def hybrid_similarity_search(self, query_node: str = None, query_text: str = None, k: int = 10, use_synonyms: bool = True) -> list[tuple[str, float, str]]:
        results = defaultdict(float)
        if query_node and query_node in self.node_mapping:
            query_embedding = self.node_embeddings[query_node].reshape(1, -1).astype('float32')
            faiss.normalize_L2(query_embedding)
            distances, indices = self.index_structural.search(query_embedding, k * 2)
            for idx, dist in zip(indices[0], distances[0]):
                node_name = self.reverse_mapping[idx]
                if node_name != query_node:
                    results[node_name] += (1 - dist) * self.structure_weight
        if query_text:
            terms = re.split(r'\s+and\s+|\s+', query_text.lower())
            terms = [term.strip() for term in terms if term.strip()]
            for term in terms:
                term_results = self._semantic_search(term, k * 2)
                for node_name, semantic_score in term_results:
                    node_type = self.G.nodes[node_name].get('type', 'Unknown')
                    type_weight = 1.0
                    if 'Uses' in node_name and term in self.medical_synonyms.get('headache', []):
                        type_weight = 1.5
                    elif 'Form' in node_name and term in self.medical_synonyms.get('tablet', []):
                        type_weight = 1.5
                    results[node_name] += semantic_score * self.text_weight * type_weight
        if use_synonyms and query_node:
            for synonym in self.synonym_lookup.get(query_node.lower(), []):
                synonym_nodes = [n for n in self.G.nodes() if synonym in n.lower()]
                for node in synonym_nodes:
                    results[node] += self.synonym_weight
        final_results = [(node, score, self.G.nodes[node].get('type', 'Unknown')) for node, score in results.items()]
        final_results.sort(key=lambda x: x[1], reverse=True)
        return final_results[:k]

    def find_drugs_smart(self, query: str, k: int = 10) -> list[tuple[str, float, list[str]]]:
        exact_node = query if query in self.G.nodes() or ':' in query else None
        drug_results = defaultdict(lambda: [0.0, []])
        terms = re.split(r'\s+and\s+|\s+', query.lower())
        terms = [term.strip() for term in terms if term.strip()]

        if exact_node:
            similar_nodes = self.hybrid_similarity_search(query_node=exact_node, k=k * 3)
            for node_name, score, node_type in similar_nodes:
                if node_type == 'BrandName':
                    drug_results[node_name][0] += score
                    drug_results[node_name][1].append(f"Direct structural match for '{query}'")
                else:
                    for neighbor in self.G.neighbors(node_name):
                        if self.G.nodes[neighbor].get('type') == 'BrandName':
                            drug_results[neighbor][0] += score * 0.8
                            drug_results[neighbor][1].append(f"Connected via {node_name}")

        for term in terms:
            term_results = self._semantic_search(term, k=k * 2)
            is_form_term = term in self.medical_synonyms.get('tablet', []) or term in self.medical_synonyms.get('syrup', [])
            term_weight = 1.5 if is_form_term else 1.0
            for node_name, semantic_score in term_results:
                node_type = self.G.nodes[node_name].get('type', 'Unknown')
                type_weight = 1.5 if (node_type == 'Uses' and term in self.medical_synonyms.get('headache', [])) else term_weight
                if node_type == 'BrandName':
                    drug_results[node_name][0] += semantic_score * self.text_weight * type_weight
                    drug_results[node_name][1].append(f"Semantic similarity to '{term}'")
                else:
                    for neighbor in self.G.neighbors(node_name):
                        if self.G.nodes[neighbor].get('type') == 'BrandName':
                            drug_results[neighbor][0] += semantic_score * self.text_weight * type_weight * 0.8
                            drug_results[neighbor][1].append(f"Semantically similar to {node_name} ({term})")

        if len(terms) > 1:
            valid_drugs = set()
            for term in terms:
                term_drugs = set([drug for drug, (score, _) in drug_results.items() if any(term in reason.lower() for reason in drug_results[drug][1])])
                valid_drugs = valid_drugs & term_drugs if valid_drugs else term_drugs
            drug_results = {drug: data for drug, data in drug_results.items() if drug in valid_drugs}

        final_results = [(drug, score, reasons) for drug, (score, reasons) in drug_results.items()]
        final_results.sort(key=lambda x: x[1], reverse=True)
        return final_results[:k]

    def get_drug_info(self, drug_name: str) -> dict[str, list[str]]:
        if drug_name not in self.G or self.G.nodes[drug_name].get('type') != 'BrandName':
            return {}
        info = defaultdict(list)
        for neighbor in self.G.neighbors(drug_name):
            if ':' in neighbor:
                attr_type, attr_value = neighbor.split(':', 1)
                info[attr_type].append(attr_value)
        return dict(info)

    def save_model(self, output_dir: str = 'hybrid_model_output'):
        os.makedirs(output_dir, exist_ok=True)
        with open(os.path.join(output_dir, 'knowledge_graph.gpickle'), 'wb') as f:
            pickle.dump(self.G, f, pickle.HIGHEST_PROTOCOL)
        with open(os.path.join(output_dir, 'structural_embeddings.pkl'), 'wb') as f:
            pickle.dump(self.node_embeddings, f)
        with open(os.path.join(output_dir, 'text_embeddings.pkl'), 'wb') as f:
            pickle.dump(self.text_embeddings, f)
        with open(os.path.join(output_dir, 'node_texts.json'), 'w') as f:
            json.dump(self.node_texts, f)
        with open(os.path.join(output_dir, 'synonyms.json'), 'w') as f:
            json.dump(self.synonyms, f)
        with open(os.path.join(output_dir, 'node_mappings.json'), 'w') as f:
            json.dump({
                'node_mapping': {str(k): v for k, v in self.node_mapping.items()},
                'reverse_mapping': {str(k): v for k, v in self.reverse_mapping.items()}
            }, f)
        logger.info(f"Hybrid model saved to {output_dir}")

    def load_model(self, input_dir: str = 'hybrid_model_output'):
        try:
            with open(os.path.join(input_dir, 'knowledge_graph.gpickle'), 'rb') as f:
                self.G = pickle.load(f)
            with open(os.path.join(input_dir, 'structural_embeddings.pkl'), 'rb') as f:
                self.node_embeddings = pickle.load(f)
            with open(os.path.join(input_dir, 'text_embeddings.pkl'), 'rb') as f:
                self.text_embeddings = pickle.load(f)
            with open(os.path.join(input_dir, 'node_texts.json'), 'r') as f:
                self.node_texts = json.load(f)
            with open(os.path.join(input_dir, 'synonyms.json'), 'r') as f:
                self.synonyms = json.load(f)
            with open(os.path.join(input_dir, 'node_mappings.json'), 'r') as f:
                mappings = json.load(f)
                self.node_mapping = {k: int(v) for k, v in mappings['node_mapping'].items()}
                self.reverse_mapping = {int(k): v for k, v in mappings['reverse_mapping'].items()}
            self.build_faiss_indices()
            logger.info(f"Hybrid model loaded from {input_dir}")
        except Exception as e:
            logger.error(f"Error loading model: {str(e)}. Rebuilding model...")
            self.preprocess_data()
            self.build_graph()
            self.generate_embeddings()
            self.build_faiss_indices()
            self.save_model(input_dir)
            logger.info(f"Rebuilt and saved to {input_dir}")