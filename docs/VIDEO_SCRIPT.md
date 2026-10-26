# Video script: under 5 minutes (aim for 3:30–4:00)

Record on Windows with **OBS Studio** (free), or with the Xbox Game Bar (Win+G) for screen capture plus the phone camera for the intro. Upload to YouTube as **Unlisted**.

| Time | On screen | You say |
|---|---|---|
| 0:00–0:20 | **You on camera** (webcam or phone selfie). The rules say the video must show the team. | "Hi, I'm [your name] from Pakistan, a solo entrant, and this is SheetSense: a grading agent for multiple-choice answer sheets, built with OpenCV 5 on AWS." |
| 0:20–0:50 | A stack of answer sheets, or the printable sheet PDF | "Teachers here mark MCQ tests by hand. One class is about 1,600 answers. Scanners are expensive, and simple phone readers just report whatever they see. A blurry photo or an erased answer quietly becomes a wrong grade. I wanted a grader that knows when it doesn't know." |
| 0:50–2:10 | **Live demo** on the deployed URL (phone or browser). Click **Use demo key**, then **Needs review**. If you printed a sheet, photograph it live instead. | "I enter the key and photograph a sheet. The agent finds the four corner markers, straightens the page and reads all 160 bubbles. It was confident on 38 questions. On questions 9 and 15 it saw an erasure, so it zoomed in and measured again to confirm the real answer. Questions 6 and 31 are faint, so it shows me a close-up and asks. I pick the answer and approve. Nothing is final until I approve it." |
| 2:10–2:35 | Click **Blurry** | "With a bad photo it doesn't guess. It tells me exactly what to fix." |
| 2:35–3:05 | Open the **trace** ("Why the agent did this"), then `docs/agent-loop.png` | "Here's the agent's trace: each OpenCV tool it called, what it saw, and why it chose the next step. Perceive, decide, act." |
| 3:05–3:30 | `docs/architecture.png`, then the CloudWatch dashboard | "It runs on AWS Lambda on Graviton arm64, behind a Function URL, deployed with one script. CloudWatch records latency, outcomes and review load. Photos are never stored." |
| 3:30–4:05 | The results table from the README | "On 7,600 held-out test questions, a single-pass reader made 64 silent mistakes. SheetSense made zero, and it caught all 40 unusable photos. The zoom step cut teacher reviews from 88 % of sheets to 47 %. The test photos are synthetic, so a real-classroom study is next." |
| 4:05–4:20 | You on camera, or the title card | "SheetSense: grading that knows when to ask. Thanks for watching." |

Tips:
- Record at 1080p.
- Zoom the browser to about 125 %.
- Close other tabs.
- Do one practice run so the Lambda is warm.
