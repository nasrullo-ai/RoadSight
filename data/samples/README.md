# Sample videos

Put the organizer's sample `.mp4` files here (not committed: large, and not ours to redistribute).

For development we also used two public traffic clips from the Roboflow `supervision` examples:

    curl -L -o data/samples/vehicles.mp4   https://media.roboflow.com/supervision/video-examples/vehicles.mp4
    curl -L -o data/samples/vehicles-2.mp4 https://media.roboflow.com/supervision/video-examples/vehicles-2.mp4

`data/dev_labels.json` holds our labels for whatever is in this folder (`tools/annotate.py`).
