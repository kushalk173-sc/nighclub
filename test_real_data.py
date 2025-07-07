#!/usr/bin/env python3
"""
Test script to verify that real data loaders work correctly.
This script tests the data loaders for pillars 1, 2, 3, 4, 5, and 6.
"""

import sys
import traceback
import torch
import numpy as np
from pathlib import Path

def test_pillar_1():
    """Test Pillar 1 (Audio-ASR) data loader."""
    print("Testing Pillar 1 (Audio-ASR)...")
    try:
        from pillars.pillar_01_audio_asr.data_loader import load_data
        from pillars.pillar_01_audio_asr.evaluate import evaluate_wer
        
        # Test data loading
        audio_data, transcripts = load_data(test_id=1, batch_size=2)
        print(f"  ✓ Audio data shape: {audio_data.shape}")
        print(f"  ✓ Number of transcripts: {len(transcripts)}")
        print(f"  ✓ Sample transcript: '{transcripts[0][:50]}...'")
        
        # Test evaluation
        mock_predictions = ["this is a test prediction"] * len(transcripts)
        wer_score = evaluate_wer(mock_predictions, transcripts)
        print(f"  ✓ WER evaluation: {wer_score:.3f}")
        
        return True
    except Exception as e:
        print(f"  ✗ Error: {e}")
        traceback.print_exc()
        return False

def test_pillar_2():
    """Test Pillar 2 (Vision) data loader."""
    print("Testing Pillar 2 (Vision)...")
    try:
        from pillars.pillar_02_vision.data_loader import load_data
        from pillars.pillar_02_vision.evaluate import evaluate
        
        # Test data loading
        image_data, labels = load_data(test_id=11, batch_size=2)
        print(f"  ✓ Image data shape: {image_data.shape}")
        print(f"  ✓ Labels shape: {labels.shape}")
        
        # Test evaluation
        mock_predictions = torch.randint(0, 1000, (len(labels),))
        accuracy = evaluate(mock_predictions, labels)
        print(f"  ✓ Accuracy evaluation: {accuracy:.3f}")
        
        return True
    except Exception as e:
        print(f"  ✗ Error: {e}")
        traceback.print_exc()
        return False

def test_pillar_3():
    """Test Pillar 3 (Time Series) data loader."""
    print("Testing Pillar 3 (Time Series)...")
    try:
        from pillars.pillar_03_regime_switching_timeseries.data_loader import load_data
        from pillars.pillar_03_regime_switching_timeseries.evaluate import evaluate
        
        # Test data loading
        time_series_data, labels = load_data(test_id=21, batch_size=2)
        print(f"  ✓ Time series data shape: {time_series_data.shape}")
        print(f"  ✓ Labels shape: {labels.shape}")
        
        # Test evaluation
        mock_predictions = torch.randn_like(labels)
        mae_score = evaluate(mock_predictions, labels, metric="mae")
        print(f"  ✓ MAE evaluation: {mae_score:.4f}")
        
        return True
    except Exception as e:
        print(f"  ✗ Error: {e}")
        traceback.print_exc()
        return False

def test_pillar_4():
    """Test Pillar 4 (Dialogue) data loader."""
    print("Testing Pillar 4 (Dialogue)...")
    try:
        from pillars.pillar_04_long_context_dialogue.data_loader import load_data
        from pillars.pillar_04_long_context_dialogue.evaluate import evaluate
        
        # Test data loading
        dialogues, labels = load_data(test_id=31, batch_size=2)
        print(f"  ✓ Number of dialogues: {len(dialogues)}")
        print(f"  ✓ Labels shape: {labels.shape}")
        print(f"  ✓ Sample dialogue: '{dialogues[0][:50]}...'")
        
        # Test evaluation
        mock_predictions = [1] * len(labels)  # All coherent
        accuracy = evaluate(mock_predictions, labels)
        print(f"  ✓ Accuracy evaluation: {accuracy:.3f}")
        
        return True
    except Exception as e:
        print(f"  ✗ Error: {e}")
        traceback.print_exc()
        return False

def test_pillar_5():
    """Test Pillar 5 (Multi-hop Reasoning) data loader."""
    print("Testing Pillar 5 (Multi-hop Reasoning)...")
    try:
        from pillars.pillar_05_multi_hop_reasoning.data_loader import load_data
        from pillars.pillar_05_multi_hop_reasoning.evaluate import evaluate
        
        # Test data loading
        qa_contexts, labels = load_data(test_id=41, batch_size=2)
        print(f"  ✓ Number of QA contexts: {len(qa_contexts)}")
        print(f"  ✓ Labels shape: {labels.shape}")
        print(f"  ✓ Sample QA: '{qa_contexts[0][:50]}...'")
        
        # Test evaluation
        mock_predictions = [1] * len(labels)  # All correct
        accuracy = evaluate(mock_predictions, labels)
        print(f"  ✓ Accuracy evaluation: {accuracy:.3f}")
        
        return True
    except Exception as e:
        print(f"  ✗ Error: {e}")
        traceback.print_exc()
        return False

def test_pillar_6():
    """Test Pillar 6 (Analogy) data loader."""
    print("Testing Pillar 6 (Analogy)...")
    try:
        from pillars.pillar_06_open_ended_analogy.data_loader import load_data
        from pillars.pillar_06_open_ended_analogy.evaluate import evaluate
        
        # Test data loading
        input_patterns, output_patterns = load_data(test_id=51, batch_size=2)
        print(f"  ✓ Input patterns shape: {input_patterns.shape}")
        print(f"  ✓ Output patterns shape: {output_patterns.shape}")
        
        # Test evaluation
        mock_predictions = torch.randn_like(output_patterns)
        mae_score = evaluate(mock_predictions, output_patterns, metric="mae")
        print(f"  ✓ MAE evaluation: {mae_score:.4f}")
        
        return True
    except Exception as e:
        print(f"  ✗ Error: {e}")
        traceback.print_exc()
        return False

def main():
    """Run all tests."""
    print("=" * 60)
    print("Testing Real Data Loaders")
    print("=" * 60)
    
    test_functions = [
        test_pillar_1,
        test_pillar_2,
        test_pillar_3,
        test_pillar_4,
        test_pillar_5,
        test_pillar_6,
    ]
    
    results = []
    for test_func in test_functions:
        result = test_func()
        results.append(result)
        print()
    
    # Summary
    print("=" * 60)
    print("Test Summary:")
    print("=" * 60)
    
    pillar_names = ["Pillar 1 (Audio-ASR)", "Pillar 2 (Vision)", "Pillar 3 (Time Series)", 
                   "Pillar 4 (Dialogue)", "Pillar 5 (Reasoning)", "Pillar 6 (Analogy)"]
    
    for i, (name, result) in enumerate(zip(pillar_names, results)):
        status = "✓ PASS" if result else "✗ FAIL"
        print(f"{name}: {status}")
    
    passed = sum(results)
    total = len(results)
    print(f"\nOverall: {passed}/{total} pillars passed")
    
    if passed == total:
        print("🎉 All real data loaders are working correctly!")
    else:
        print("⚠️  Some data loaders need attention.")
    
    return passed == total

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1) 