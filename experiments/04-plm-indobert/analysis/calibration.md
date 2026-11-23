# Calibration of answerability

na_prob is a raw logit difference (null minus best span score); sigmoid is an uncalibrated monotone map, so ECE mostly shows how far that scale is from a probability. 10 equal-width bins.

- plm_indobert_full seed13: ECE 0.1282, mean sigmoid(na_prob) 0.503, unanswerable rate 0.462, tuned threshold on the raw logit -0.508
- plm_indobert_full seed42: ECE 0.1328, mean sigmoid(na_prob) 0.483, unanswerable rate 0.462, tuned threshold on the raw logit -2.199
- plm_indobert_full seed87: ECE 0.1313, mean sigmoid(na_prob) 0.485, unanswerable rate 0.462, tuned threshold on the raw logit -3.220
