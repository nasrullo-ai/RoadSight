# Sample videos

Put the organizer's four sample videos here (not committed: ~20 GB, and not ours to redistribute):
`C3896.MP4`, `C3897.MP4`, `C3902.MP4`, `C3905.MP4` (links in the task's `Videos.pdf`).
`predictions_samples.json` at the repository root is the harness output on exactly these four files.

Two public traffic clips from the Roboflow `supervision` examples were used during development and
live in `data/dev_clips/` instead (they contain no events; `data/dev_labels.json` says so):

    curl -L -o data/dev_clips/vehicles.mp4   https://media.roboflow.com/supervision/video-examples/vehicles.mp4
    curl -L -o data/dev_clips/vehicles-2.mp4 https://media.roboflow.com/supervision/video-examples/vehicles-2.mp4
