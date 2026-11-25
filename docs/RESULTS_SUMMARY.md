# Results so far, in plain terms

Written November 24, 2026. Every number is on validation recordings from FRED, an event-camera drone dataset. The full record, with caveats, is in [FINDINGS.md](FINDINGS.md).

## The data

An event camera reports each pixel that gets brighter (an ON event) or darker (an OFF event), with a timestamp in microseconds. It never takes whole pictures.

- **Box:** a rectangle around a drone, or around a patch of busy background of the same size. Only events inside it count.
- **Window:** all the events in one box over 0.5 s. Each window is one example: drone or background.
- **Packet:** a run of consecutive events inside a window, sized so a window has at most 500 packets. Each packet records the share of ON events and the time it took to arrive.
- **Δt (gap):** the time between one packet and the next. A spinning rotor makes events arrive in quick, regular bursts, so Δt carries the rotor's rhythm.
- **Dropped packets:** packets deleted at random to imitate a camera that sees fewer events, such as a smaller, farther or hovering drone, or our lower-resolution SilkyEvCam.

Training uses five recordings, one per drone type (7,435 windows). Validation uses five other recordings (8,278 windows).

## The models

Every learned model has about 4,900 parameters and was trained 5 times with different random starts (seeds).

- **FFT + count:** no learning. Measures how strongly the box's event count repeats at one frequency, and how many events there are.
- **CNN (convolutional network):** looks at the whole window at once and slides small learned filters along it to find short patterns.
- **GRU (gated recurrent unit):** reads packets one at a time and keeps a running summary. A learned gate decides how much of the old summary to keep, so information carries across hundreds of packets. It gets Δt as an input.
- **LSTM:** an older recurrent design, like the GRU, with a separate memory store.
- **CfC (closed-form continuous-time network):** also reads packets in order, but Δt sits inside its gate, so its state moves between two learned targets depending on how much time passed. The plain version has no way to copy its summary forward.
- **CfC + memory:** a CfC with an LSTM attached to carry memory, which fixes that.

## The results

**AUC** is the chance a random drone window scores above a random background window: 0.5 is guessing, 1.0 is perfect. **Matched AUC** is the same, using only drone and background windows with nearly the same number of events, so counting events alone gets 0.50.

| Model | AUC | Matched AUC |
|---|---:|---:|
| FFT + count | 0.952 | 0.608 |
| CNN (sees counts in 0.5 ms bins, not packets) | 0.967 | 0.853 |
| GRU | **0.979** | **0.911** |
| LSTM | 0.968 | 0.879 |
| CfC | 0.960 | 0.851 |
| CfC + memory | 0.967 | 0.889 |

With packets dropped at random during testing (AUC):

| Model | Complete windows | 50% dropped | 90% dropped |
|---|---:|---:|---:|
| GRU | 0.979 | 0.822 | 0.747 |
| CfC + memory | 0.967 | 0.855 | 0.835 |
| GRU, trained with packets dropped | 0.967 | 0.957 | 0.922 |
| CfC + memory, trained with packets dropped | 0.965 | 0.959 | 0.929 |

## What it means

1. **Timing matters.** The CfC, CfC + memory and GRU all score lower when Δt is replaced with a constant (the LSTM was not tested that way). The CfC loses 0.066 matched AUC without it.
2. **On ordinary data the GRU is best,** ahead of the CfC + memory by 0.022 matched AUC in all 5 seeds, and by 0.082 once each model's learning rate is tuned. The CfC + memory ties the CNN on AUC.
3. **The CfC's design helps when timing changes and training didn't prepare for it.** Trained on complete windows, it keeps more accuracy than the GRU when packets go missing.
4. **Training with dropped packets gives the GRU the same robustness.** With that training both models are level, so the CfC's built-in timing doesn't beat simply showing a GRU varied data.
5. **So far, the claim we can defend:** a CfC degrades more gracefully than a GRU trained the ordinary way. Nothing here shows a CfC beating a well-trained GRU.

## After tuning the learning rate

The tables above used one learning rate (how big each training step is) for every model. Each model was then retrained at 0.0003, 0.003 and 0.01 as well, and kept the rate with the best matched AUC:

| Model, at its best rate | Rate | AUC | Matched AUC |
|---|---:|---:|---:|
| GRU | 0.01 | **0.990** | **0.971** |
| GRU, drop-trained | 0.01 | 0.983 | 0.956 |
| LSTM | 0.01 | 0.984 | 0.952 |
| CfC + memory, drop-trained | 0.01 | 0.975 | 0.943 |
| CfC + memory | 0.001 | 0.967 | 0.889 |
| CNN | 0.01 | 0.962 | 0.887 |
| CfC | 0.003 | 0.970 | 0.872 |

Tuning helped the GRU most: its lead over the CfC + memory grew from 0.022 to 0.082 matched AUC, in all 5 runs. Both CfCs trained on complete windows broke down one rate above their best (the CfC + memory at 0.003, the plain CfC at 0.01), while the GRU was still improving at 0.01. Validation picked the rate, so these numbers are slightly optimistic.

## Next steps

1. **Try a warm-up schedule or tighter gradient clipping** for every model, to see whether the CfC can train at the faster rates that helped the GRU.
2. **Record a DJI Mini 4K with our SilkyEvCam** once the lens arrives, to test on our own camera with a real change in event rate.
3. **Score FRED's challenging split once,** after the models stop changing. It has not been touched by any learned model.
4. **Add more training recordings** and some with birds; there are five now, and no birds.
