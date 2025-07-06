# Pillar 1: Instant Domain Shift (Audio-ASR)

This pillar evaluates the model's ability to rapidly adapt to different audio domains for Automatic Speech Recognition (ASR).

## Tests

The 10 tests in this pillar expose the model to a variety of conditions, including:
- Different microphones (studio, laptop, smartphone)
- Different acoustic environments (car, cafe, outdoors)
- Different speaker accents (US, UK, Irish, etc.)

## Metric

The primary metric is **Word Error Rate (WER)**. For the first test, WER is calculated on the audio after the first 3 seconds to specifically measure rapid adaptation. 