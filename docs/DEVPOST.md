# Devpost submission copy (paste field by field)

## Project name
SheetSense

## Elevator pitch (≤200 chars)
Grade MCQ answer sheets from a phone photo. An OpenCV 5 agent on AWS Lambda re-checks what it can't see, asks for retakes, and lets the teacher approve every grade.

## About the project (Markdown)

## Inspiration
Teachers mark multiple-choice tests by hand. One class of 40 students with 40 questions is 1,600 answers to check, and that kind of repetitive work is where mistakes slip in. OMR scanners need special hardware most schools and tuition centres in Pakistan don't have. Simple phone readers make one pass and report whatever they saw, so a blurry photo, a glare spot or an erased answer quietly becomes a wrong grade. We wanted a grader that **knows when it doesn't know**.

## What it does
- **Print** the free SheetSense A4 sheet: 40 questions A–D, with four ArUco corner markers.
- **Enter** the answer key, or photograph a filled key sheet.
- **Photograph** each student's sheet with any phone. An agent built on OpenCV 5 runs on AWS Lambda (arm64/Graviton) and does three things:
  - finds and straightens the page at any angle, even upside down
  - checks that the photo is sharp, close enough and free of glare
  - reads all 160 bubbles
- **Returns one of three outcomes:**
  - **Ready:** every answer was read confidently.
  - **Needs review:** the teacher sees a close-up of each uncertain question with the reason ("faint mark", "glare over this row") and taps the answer.
  - **Retake:** a specific instruction, such as "top-right corner square not visible" or "photo is blurry".
- **No grade is final until the teacher approves it.** Approved grades go to a class list with CSV export.

## How we built it
- **Perception (OpenCV 5.0.0):**
  - `ArucoDetector` → RANSAC `findHomography` to a millimetre page frame, with a reprojection-error check
  - escalating fallbacks: CLAHE → illumination flattening → downscale → page outline (`Canny`/`findContours`/`approxPolyDP`) + `matchTemplate` over 4 rotations
  - sharpness from `Laplacian`
  - glare detection that uses the **printed bubble rings as a built-in calibration target**
  - per-sheet calibrated ink darkness after dividing out lighting
- **Agent:** a perception → decision → action policy. What it sees changes what it does next:
  - it escalates detectors only when needed
  - it asks for a specific retake
  - it **zooms**: re-warps just the uncertain row from the original photo at 14 px/mm and adds a second measurement (Otsu ink coverage) that tells an erasure smudge from a real mark
  - it sends anything still uncertain to the teacher

  Every step is written to a trace shown in the app and logged to CloudWatch.
- **AWS:**
  - Lambda arm64 (Graviton), Python 3.14, behind a Function URL
  - a one-command idempotent `deploy.sh` (pinned aarch64 wheels → S3 → Lambda)
  - an IAM role that can only write logs
  - CloudWatch Embedded Metric Format logs and a dashboard (latency, outcomes, review load, tool calls)
  - photos are never stored
- **Evaluation:** a seeded generator renders the real template, fills it with pen/pencil marks, erasures, faint and double marks, then simulates phone photos (perspective, shadows, glare, blur, noise, JPEG). Thresholds were tuned on a dev set and frozen; results come from a separate held-out test set.

## Results (held-out synthetic test set: 230 photos, 7,600 questions)
| | Single-pass baseline | SheetSense agent |
|---|---|---|
| Silent wrong answers | 64 | **0** (95 % upper bound ≈ 0.41 per 1,000) |
| Decided automatically | 98.9 % | 95.5 % |
| Sent to the teacher with a close-up | 0 % | 1.9 % (0.76 per sheet) |
| Unusable photos caught | 21 / 40 | **40 / 40** |
| Good photos sent back for a retake | 2 / 190 | 5 / 190 |

Without the zoom step, 88 % of sheets needed a teacher check (mostly erasures). With zoom only 47 % do, and accuracy is unchanged.

## Challenges we ran into
- **Glare:** a glare spot can erase one bubble while the rest of its row looks fine. Checking each printed ring separately fixed it.
- **Faint marks:** our first zoom step confidently called them blank. We rewrote it so the agent can only resolve a question when two independent measurements agree; otherwise it asks.

## Accomplishments that we're proud of
- **Zero silent errors** on 7,600 held-out questions.
- **Every unusable photo caught**, each with a retake instruction a person can act on.
- **The same results on arm64 Lambda and x86.**

## What we learned
Most of the value is in deciding *when not to answer*. Active perception, meaning going back to the original pixels for only the hard cases, cut the teacher's review work by more than two-thirds.

## What's next
- A study with real classroom photos (the evaluation code already accepts any labelled folder)
- Student-ID bubbles
- Configurable layouts
- Urdu interface text
- Offline batching for low-connectivity schools

## Built with
opencv, python, aws-lambda, aws-graviton, amazon-cloudwatch, amazon-s3, aws-iam, numpy, javascript, html5

## Try it out
- Live app: https://5omclyghhagcueiorjtjc4ieou0bfdzg.lambda-url.ap-south-1.on.aws/
- Example with review: https://5omclyghhagcueiorjtjc4ieou0bfdzg.lambda-url.ap-south-1.on.aws/?sample=review
- Code: <REPO_URL>

## Video
<YOUTUBE_URL> (unlisted is fine)
