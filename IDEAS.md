# FPGA project choices

| Idea | Live demo | Why the FPGA matters | Main dependency |
| --- | --- | --- | --- |
| **Neural Reflex** (recommended) | A new participant calibrates their muscle signals and controls a gripper | Streaming, fixed-point feature extraction and inference with measurable jitter | Axo band and raw EMG API |
| Private video reflex | A camera sends only detected motion or events to an AI model | Video preprocessing at line rate can cut host traffic | A reliable video-to-FPGA path |
| Physical AI safety coprocessor | Hardware blocks commands outside a robot's safe workspace | Independent, deterministic watchdog and bounds checks | Robot command access |

## Why Neural Reflex wins

It combines a visually clear live demonstration, subject-specific learning, real FPGA work, and the hackathon's Axo challenge. The XEM7310's USB 3 FrontPanel link gives us a practical host-to-FPGA path. The ambitious extension is two-person collaborative control using one band and FPGA per participant, with a Medra arm performing the physical task. We should only pursue that extension after the one-person virtual demo works.

## First decisions at the venue

Reproduce the owner's working laptop-to-XEM7310 USB-C setup and locate its current bitstream build flow. Ask Axo for the raw EMG Mac API and two bands; ask Medra whether a safe robot command API is available. Find the variant label on the back of the XEM7310 and the teammate's board model. If any critical dependency is unavailable, preserve the single-board FPGA signal-processing work and choose the best available live input.
