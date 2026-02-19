#!/usr/bin/env python3
"""
Quick Token Viewer for SentencePiece Tokenizer
A simple script to quickly view all tokens in your SentencePiece tokenizer.
"""

import os
import sentencepiece as spm

def view_all_tokens(model_path="sp_model.model", max_tokens=None):
    """View all tokens in the SentencePiece model"""
    
    if not os.path.exists(model_path):
        print(f"❌ Model not found at {model_path}")
        return
    
    # Load the model
    sp = spm.SentencePieceProcessor()
    sp.load(model_path)
    
    vocab_size = sp.get_piece_size()
    print(f"🔍 SentencePiece Tokenizer Analysis")
    print(f"📊 Vocabulary size: {vocab_size}")
    print(f"📁 Model: {model_path}")
    print("="*80)
    
    # Display tokens by category
    control_tokens = []
    byte_tokens = []
    word_tokens = []
    subword_tokens = []
    
    for i in range(vocab_size):
        token = sp.id_to_piece(i)
        score = sp.get_score(i)
        
        if token.startswith('<') and token.endswith('>'):
            if token.startswith('<0x') and token.endswith('>'):
                byte_tokens.append((i, token, score))
            else:
                control_tokens.append((i, token, score))
        elif token.startswith('▁'):
            word_tokens.append((i, token, score))
        else:
            subword_tokens.append((i, token, score))
    
    # Display control tokens
    print(f"\n🔧 CONTROL TOKENS ({len(control_tokens)}):")
    print("-" * 40)
    for token_id, token, score in control_tokens:
        print(f"  {token_id:4d}: '{token}' (score: {score:6.2f})")
    
    # Display byte tokens (first few)
    print(f"\n🔢 BYTE TOKENS ({len(byte_tokens)} total, showing first 20):")
    print("-" * 40)
    for token_id, token, score in byte_tokens[:20]:
        print(f"  {token_id:4d}: '{token}' (score: {score:6.2f})")
    if len(byte_tokens) > 20:
        print(f"  ... and {len(byte_tokens) - 20} more byte tokens")
    
    # Display word tokens (first 50)
    print(f"\n📝 WORD TOKENS ({len(word_tokens)} total, showing first 50):")
    print("-" * 40)
    for token_id, token, score in word_tokens[:50]:
        print(f"  {token_id:4d}: '{token}' (score: {score:6.2f})")
    if len(word_tokens) > 50:
        print(f"  ... and {len(word_tokens) - 50} more word tokens")
    
    # Display subword tokens (first 50)
    print(f"\n🔤 SUBWORD TOKENS ({len(subword_tokens)} total, showing first 50):")
    print("-" * 40)
    for token_id, token, score in subword_tokens[:50]:
        print(f"  {token_id:4d}: '{token}' (score: {score:6.2f})")
    if len(subword_tokens) > 50:
        print(f"  ... and {len(subword_tokens) - 50} more subword tokens")
    
    # Summary statistics
    print(f"\n📈 SUMMARY:")
    print("-" * 40)
    print(f"Total vocabulary: {vocab_size}")
    print(f"Control tokens: {len(control_tokens)}")
    print(f"Byte tokens: {len(byte_tokens)}")
    print(f"Word tokens: {len(word_tokens)}")
    print(f"Subword tokens: {len(subword_tokens)}")
    
    # Show some example tokenizations
    print(f"\n🔍 EXAMPLE TOKENIZATIONS:")
    print("-" * 40)
    
    example_texts = [
        "Hello world!",
        "The quick brown fox jumps over the lazy dog.",
        "Python programming is fun.",
        "1234567890",
        "Special characters: @#$%^&*()"
    ]
    
    for text in example_texts:
        pieces = sp.encode_as_pieces(text)
        ids = sp.encode_as_ids(text)
        print(f"\nText: {repr(text)}")
        print(f"Tokens: {pieces}")
        print(f"IDs: {ids}")
        print(f"Token count: {len(pieces)}")

def export_tokens_to_file(model_path="sp_model.model", output_file="all_tokens.txt"):
    """Export all tokens to a text file"""
    
    if not os.path.exists(model_path):
        print(f"❌ Model not found at {model_path}")
        return
    
    # Load the model
    sp = spm.SentencePieceProcessor()
    sp.load(model_path)
    
    vocab_size = sp.get_piece_size()
    
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(f"SentencePiece Tokenizer Vocabulary\n")
        f.write(f"Model: {model_path}\n")
        f.write(f"Vocabulary size: {vocab_size}\n")
        f.write("="*80 + "\n\n")
        
        # Write all tokens
        for i in range(vocab_size):
            token = sp.id_to_piece(i)
            score = sp.get_score(i)
            
            # Categorize token
            if token.startswith('<') and token.endswith('>'):
                if token.startswith('<0x') and token.endswith('>'):
                    category = "BYTE"
                else:
                    category = "CONTROL"
            elif token.startswith('▁'):
                category = "WORD"
            else:
                category = "SUBWORD"
            
            f.write(f"{i:4d}: '{token}' (score: {score:6.2f}, type: {category})\n")
    
    print(f"✅ All tokens exported to {output_file}")

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == "export":
        export_tokens_to_file()
    else:
        view_all_tokens()
