#!/usr/bin/env python3
"""
Token Visualizer for SentencePiece Tokenizer
A comprehensive tool to explore and visualize all tokens in your SentencePiece tokenizer.
"""

import os
import pickle
import json
from collections import Counter, defaultdict
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from mingpt.sentencepiece_tokenizer import SentencePieceTokenizer
import sentencepiece as spm

class TokenVisualizer:
    """Comprehensive token visualization and analysis tool"""
    
    def __init__(self, model_path="sp_model.model"):
        """Initialize the visualizer with a SentencePiece model"""
        self.model_path = model_path
        self.sp = None
        self.vocab = {}
        self.token_frequencies = {}
        
        if os.path.exists(model_path):
            self.sp = spm.SentencePieceProcessor()
            self.sp.load(model_path)
            print(f"✅ Loaded SentencePiece model from {model_path}")
            print(f"📊 Vocabulary size: {self.sp.get_piece_size()}")
            self._load_vocabulary()
        else:
            print(f"❌ Model not found at {model_path}")
    
    def _load_vocabulary(self):
        """Load the vocabulary and token information"""
        vocab_size = self.sp.get_piece_size()
        
        for i in range(vocab_size):
            token = self.sp.id_to_piece(i)
            score = self.sp.get_score(i)
            self.vocab[i] = {
                'token': token,
                'score': score,
                'is_control': token.startswith('<') and token.endswith('>'),
                'is_byte': token.startswith('<0x') and token.endswith('>'),
                'length': len(token),
                'has_underscore': token.startswith('▁'),
                'is_alpha': token.replace('▁', '').isalpha(),
                'is_digit': token.replace('▁', '').isdigit(),
                'is_punctuation': not token.replace('▁', '').isalnum() and not token.startswith('<')
            }
    
    def show_vocabulary_summary(self):
        """Display a comprehensive summary of the vocabulary"""
        print("\n" + "="*60)
        print("📚 VOCABULARY SUMMARY")
        print("="*60)
        
        vocab_size = len(self.vocab)
        control_tokens = sum(1 for info in self.vocab.values() if info['is_control'])
        byte_tokens = sum(1 for info in self.vocab.values() if info['is_byte'])
        word_tokens = sum(1 for info in self.vocab.values() if info['has_underscore'])
        subword_tokens = vocab_size - control_tokens - byte_tokens - word_tokens
        
        print(f"Total vocabulary size: {vocab_size}")
        print(f"Control tokens: {control_tokens}")
        print(f"Byte tokens: {byte_tokens}")
        print(f"Word tokens (start with ▁): {word_tokens}")
        print(f"Subword tokens: {subword_tokens}")
        
        # Show control tokens
        control_tokens_list = [info['token'] for info in self.vocab.values() if info['is_control']]
        print(f"\n🔧 Control tokens: {', '.join(control_tokens_list)}")
        
        # Show some example tokens
        print(f"\n📝 Example word tokens:")
        word_examples = [info['token'] for info in self.vocab.values() if info['has_underscore']][:10]
        print(f"   {', '.join(word_examples)}")
        
        print(f"\n🔤 Example subword tokens:")
        subword_examples = [info['token'] for info in self.vocab.values() 
                          if not info['is_control'] and not info['is_byte'] and not info['has_underscore']][:10]
        print(f"   {', '.join(subword_examples)}")
    
    def analyze_text_tokens(self, text, show_details=True):
        """Analyze how a specific text gets tokenized"""
        if not self.sp:
            print("❌ No tokenizer loaded")
            return
        
        print(f"\n🔍 TOKENIZING TEXT:")
        print(f"Text: {repr(text)}")
        print(f"Length: {len(text)} characters")
        
        # Get tokenization
        pieces = self.sp.encode_as_pieces(text)
        ids = self.sp.encode_as_ids(text)
        
        print(f"\n📊 Tokenization Results:")
        print(f"Number of tokens: {len(pieces)}")
        print(f"Compression ratio: {len(text) / len(pieces):.2f} chars per token")
        
        if show_details:
            print(f"\n🔤 Detailed tokenization:")
            for i, (piece, token_id) in enumerate(zip(pieces, ids)):
                token_info = self.vocab[token_id]
                print(f"  {i:3d}: '{piece}' (ID: {token_id:4d}, Score: {token_info['score']:6.2f})")
        
        return pieces, ids
    
    def show_token_statistics(self, top_n=50):
        """Show statistics about token usage and characteristics"""
        print(f"\n📈 TOKEN STATISTICS")
        print("="*60)
        
        # Token length distribution
        lengths = [info['length'] for info in self.vocab.values()]
        print(f"Token length statistics:")
        print(f"  Average length: {sum(lengths) / len(lengths):.2f}")
        print(f"  Min length: {min(lengths)}")
        print(f"  Max length: {max(lengths)}")
        
        # Score distribution
        scores = [info['score'] for info in self.vocab.values()]
        print(f"\nScore statistics:")
        print(f"  Average score: {sum(scores) / len(scores):.2f}")
        print(f"  Min score: {min(scores)}")
        print(f"  Max score: {max(scores)}")
        
        # Top tokens by score (excluding control tokens)
        regular_tokens = [(i, info) for i, info in self.vocab.items() 
                         if not info['is_control'] and not info['is_byte']]
        top_by_score = sorted(regular_tokens, key=lambda x: x[1]['score'], reverse=True)[:top_n]
        
        print(f"\n🏆 Top {top_n} tokens by score:")
        for i, (token_id, info) in enumerate(top_by_score):
            print(f"  {i+1:3d}: '{info['token']}' (Score: {info['score']:6.2f})")
    
    def search_tokens(self, query, max_results=20):
        """Search for tokens containing a specific pattern"""
        print(f"\n🔍 SEARCHING FOR TOKENS CONTAINING '{query}'")
        print("="*60)
        
        matches = []
        for token_id, info in self.vocab.items():
            if query.lower() in info['token'].lower():
                matches.append((token_id, info))
        
        matches.sort(key=lambda x: x[1]['score'], reverse=True)
        
        if matches:
            print(f"Found {len(matches)} matching tokens:")
            for i, (token_id, info) in enumerate(matches[:max_results]):
                print(f"  {i+1:3d}: '{info['token']}' (ID: {token_id:4d}, Score: {info['score']:6.2f})")
            
            if len(matches) > max_results:
                print(f"  ... and {len(matches) - max_results} more")
        else:
            print("No tokens found matching the query.")
    
    def visualize_token_distribution(self, save_path=None):
        """Create visualizations of token characteristics"""
        print(f"\n📊 CREATING TOKEN DISTRIBUTION VISUALIZATIONS")
        
        # Prepare data
        data = []
        for token_id, info in self.vocab.items():
            if not info['is_control'] and not info['is_byte']:
                data.append({
                    'token': info['token'],
                    'score': info['score'],
                    'length': info['length'],
                    'type': 'word' if info['has_underscore'] else 'subword'
                })
        
        df = pd.DataFrame(data)
        
        # Create subplots
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('SentencePiece Tokenizer Analysis', fontsize=16)
        
        # 1. Token length distribution
        axes[0, 0].hist(df['length'], bins=30, alpha=0.7, color='skyblue', edgecolor='black')
        axes[0, 0].set_title('Token Length Distribution')
        axes[0, 0].set_xlabel('Token Length')
        axes[0, 0].set_ylabel('Frequency')
        axes[0, 0].grid(True, alpha=0.3)
        
        # 2. Score distribution
        axes[0, 1].hist(df['score'], bins=50, alpha=0.7, color='lightgreen', edgecolor='black')
        axes[0, 1].set_title('Token Score Distribution')
        axes[0, 1].set_xlabel('Token Score')
        axes[0, 1].set_ylabel('Frequency')
        axes[0, 1].grid(True, alpha=0.3)
        
        # 3. Token type distribution
        type_counts = df['type'].value_counts()
        axes[1, 0].pie(type_counts.values, labels=type_counts.index, autopct='%1.1f%%', 
                      colors=['lightcoral', 'lightblue'])
        axes[1, 0].set_title('Token Type Distribution')
        
        # 4. Score vs Length scatter plot
        axes[1, 1].scatter(df['length'], df['score'], alpha=0.6, s=20)
        axes[1, 1].set_title('Token Score vs Length')
        axes[1, 1].set_xlabel('Token Length')
        axes[1, 1].set_ylabel('Token Score')
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"📊 Visualization saved to {save_path}")
        
        plt.show()
    
    def export_vocabulary(self, output_path="vocabulary_analysis.json"):
        """Export the complete vocabulary analysis to JSON"""
        print(f"\n💾 EXPORTING VOCABULARY ANALYSIS")
        
        export_data = {
            'model_path': self.model_path,
            'vocab_size': len(self.vocab),
            'vocabulary': {}
        }
        
        for token_id, info in self.vocab.items():
            export_data['vocabulary'][str(token_id)] = {
                'token': info['token'],
                'score': info['score'],
                'is_control': info['is_control'],
                'is_byte': info['is_byte'],
                'length': info['length'],
                'has_underscore': info['has_underscore'],
                'is_alpha': info['is_alpha'],
                'is_digit': info['is_digit'],
                'is_punctuation': info['is_punctuation']
            }
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(export_data, f, indent=2, ensure_ascii=False)
        
        print(f"✅ Vocabulary analysis exported to {output_path}")
    
    def interactive_explorer(self):
        """Interactive command-line explorer for the vocabulary"""
        print(f"\n🎮 INTERACTIVE VOCABULARY EXPLORER")
        print("="*60)
        print("Commands:")
        print("  summary    - Show vocabulary summary")
        print("  stats      - Show token statistics")
        print("  search <q> - Search for tokens containing query")
        print("  tokenize <text> - Tokenize specific text")
        print("  export     - Export vocabulary analysis")
        print("  visualize  - Create visualizations")
        print("  quit       - Exit explorer")
        print("="*60)
        
        while True:
            try:
                command = input("\n🔍 Enter command: ").strip().lower()
                
                if command == 'quit' or command == 'exit':
                    print("👋 Goodbye!")
                    break
                elif command == 'summary':
                    self.show_vocabulary_summary()
                elif command == 'stats':
                    self.show_token_statistics()
                elif command == 'export':
                    self.export_vocabulary()
                elif command == 'visualize':
                    self.visualize_token_distribution()
                elif command.startswith('search '):
                    query = command[7:].strip()
                    if query:
                        self.search_tokens(query)
                    else:
                        print("❌ Please provide a search query")
                elif command.startswith('tokenize '):
                    text = command[9:].strip()
                    if text:
                        self.analyze_text_tokens(text)
                    else:
                        print("❌ Please provide text to tokenize")
                else:
                    print("❌ Unknown command. Type 'quit' to exit.")
                    
            except KeyboardInterrupt:
                print("\n👋 Goodbye!")
                break
            except Exception as e:
                print(f"❌ Error: {e}")


def main():
    """Main function to run the token visualizer"""
    print("🔍 SentencePiece Token Visualizer")
    print("="*50)
    
    # Initialize visualizer
    visualizer = TokenVisualizer()
    
    if not visualizer.sp:
        print("❌ Could not load tokenizer. Please ensure sp_model.model exists.")
        return
    
    # Show initial summary
    visualizer.show_vocabulary_summary()
    
    # Start interactive explorer
    visualizer.interactive_explorer()


if __name__ == "__main__":
    main()
