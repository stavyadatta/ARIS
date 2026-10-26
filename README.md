# ARIS: Agentic and Relationship Intelligence System

ARIS is a neuro-symbolic agentic framework for social robots. It combines a
large language model with multimodal perception, a persistent **Social World
Model** (a knowledge graph of the people the robot has met and how they relate
to each other), and a **retrieval-augmented (RAG) speech pipeline**, so a robot
can hold long, personalised conversations, recognise returning users, see, and
move. This repository holds the reference implementation for the
SoftBank Pepper robot.

## Architecture

ARIS follows an **edge–client–server** design:

```
 ┌────────────┐   qi (NAOqi)   ┌──────────────────┐     gRPC      ┌──────────────────────────────────────┐
 │   Edge     │ ◄────────────► │      Client      │ ◄───────────► │                Server                │
 │ Pepper 1.8 │  audio, video, │ pepper_client/   │  audio+frame  │ Whisper ─► Reasoner ─► Executor ─► APIs│
 │            │  speech, motion│ client.py        │  text / JSON  │              │                       │
 └────────────┘                └──────────────────┘               │   Face ReID ─┤   Social World Model  │
                                                                  │              └── Neo4j (graph+vector)│
                                                                  └──────────────────────────────────────┘
```

* **Edge**: Pepper captures audio and video and carries out speech and motion.
* **Client** (`pepper_client/client.py`): talks to the robot through `qi`.
  It streams camera frames, records each utterance with a front-microphone
  energy gate, sends the audio and the latest frame to the server, then speaks
  the streamed reply (with body-talk animations) or runs the JSON movement
  command it receives.
* **Server** (`server/`): a gRPC endpoint that
  1. transcribes the audio with **Whisper large-v3**,
  2. re-identifies the speaker with **InsightFace**, using a majority vote over
     recent camera frames to assign a `face_id`,
  3. runs the **Reasoner**, an LLM that reads the transcript and picks an API
     from the plug-and-play API library (`server/apis/__init__.py`),
  4. runs the **Executor**, which calls that API and streams the result back.

Available APIs:

| Reasoner state | API | What it does |
|---|---|---|
| `speak` | Speech API (`apis/speaking.py`) | RAG-based conversational reply |
| `vision` | Vision API (`apis/person_attr.py`) | VLM answers a visual question about the cropped person |
| `custom movement` | Physical embodiment (`apis/movement/custom_movement.py`) | LLM produces joint angles, e.g. a high-five or handshake |
| `standard movement` | Physical embodiment (`apis/movement/standard_movement.py`) | Pre-built dance animations |
| `silent` / `bad input` | `apis/silent.py`, `apis/bad_input.py` | Stay quiet / ignore speech not meant for the robot |

### Social World Model

Each user is a `Person` node in Neo4j holding the name, `face_id` and
attributes such as interests and occupation. Relationships between people
(friend, parent, etc.) are graph edges. After every turn, an LLM pulls out
relationship and attribute cues (`core_api/relationship_checker`,
`core_api/attribute_finder`) and turns them into Cypher updates. When a user
says their name, it is matched against existing names by Levenshtein distance,
so ASR errors in names do not create duplicate people.

### RAG speech pipeline

Each user utterance and robot reply is stored as a `Message` node linked to its
`Person`. Messages are embedded with OpenAI `text-embedding-3-small`
(1536-d) and indexed in a Neo4j HNSW vector index (`message_embeddings`, cosine
similarity), which the server creates on first start. For each new input, the
Speech API retrieves:

* the user's messages among the 20 nearest neighbours of the input embedding,
  each with its preceding and following message, and
* the user's most recent conversation history.

These are merged into one chronological chain with duplicates removed and
passed to the LLM together with the user's name, attributes and relationships
(see `get_person_messages` in `server/utils/neo4j_db/database.py`).

## Repository layout

```
.
├── server/                 # ARIS server (gRPC, reasoner, executor, APIs, Social World Model)
│   ├── main.py             # entry point (gRPC on :50051, frame viewer on :8003)
│   ├── media_manager/      # gRPC MediaService implementation
│   ├── reasoner/           # LLM-based API selection
│   ├── executor/           # dispatches the reasoner's decision to an API
│   ├── apis/               # plug-and-play API library
│   ├── core_api/           # Whisper, face ReID, YOLO, LLM clients, relationship/attribute extraction
│   └── utils/              # Neo4j access layer, data objects
├── pepper_client/
│   ├── client.py           # single-file Pepper client
│   ├── requirements.txt
│   └── qi-3.1.4-...aarch64.whl   # NAOqi Python SDK for ARM64 (e.g. Raspberry Pi 5)
├── grpc_communication/     # grpc.proto and generated Python stubs
├── image_viewer.py         # debug page showing the frame the server last processed
├── docker/                 # Dockerfiles for server and client
├── docker-compose.yml          # server stack (Neo4j + ARIS server)
├── docker-compose.client.yml   # Pepper client
└── .env.example
```

## Requirements

**Server host**
* Linux x86_64 with an NVIDIA GPU (≥ 12 GB VRAM recommended for Whisper
  large-v3; developed on an RTX 4090)
* Docker ≥ 24 with the Compose plugin
* [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html)
* An OpenAI API key (reasoner, speech, vision, embeddings). An xAI Grok key is
  optional; with it, the reasoner, speech and vision calls retry on Grok when
  an OpenAI call fails.

**Client host**
* Any Linux machine on the same network as Pepper (developed on a Raspberry
  Pi 5 running 64-bit OS), amd64 or arm64
* Docker, or Python 3.10

**Robot**
* SoftBank Pepper (NAOqi 2.5 / Pepper 1.8), reachable over the network.

## Quick start (Docker)

### 1. Configure

```bash
git clone <this-repo> aris && cd aris
cp .env.example .env
# edit .env: set OPENAI_API_KEY, NEO4J_PASSWORD (and optionally GROK_API_KEY)
```

### 2. Start the server (GPU machine)

```bash
docker compose up --build -d          # starts Neo4j and the ARIS server
docker compose logs -f server         # wait for "gRPC server running on port 50051..."
```

On first start the server downloads Whisper large-v3 (~3 GB) and InsightFace
`buffalo_l` into named Docker volumes, so later starts are fast. The YOLOv8n
weights are baked into the image at build time.

| Port | Service |
|---|---|
| `50051` | ARIS gRPC endpoint (client connects here) |
| `127.0.0.1:7474` / `127.0.0.1:7687` | Neo4j browser / Bolt (inspect the Social World Model) |
| `127.0.0.1:8003` | Live view of the last frame the server processed (debug) |

Neo4j and the frame viewer listen on localhost only because they expose
personal data. To reach them from another machine, use an SSH tunnel, e.g.
`ssh -L 7474:localhost:7474 -L 7687:localhost:7687 <server>`.

Enrolled face embeddings are stored in `./database/face_db` on the host.

### 3. Start the client (robot-side machine)

Clone the repository on the machine next to the robot. Copy `.env.example`
to `.env` and set `PEPPER_IP`, `ARIS_SERVER` (the server's `host:50051`) and
`CLIENT_LISTEN_URL`. The server keys are not needed here. Then run:

```bash
docker compose -f docker-compose.client.yml up --build
```

The client uses host networking because Pepper calls back into the client over
`qi` to deliver microphone audio. If the machine has several network
interfaces, set `CLIENT_LISTEN_URL=tcp://<client-ip-on-robot-network>:9559`.

### 4. Interact

* Pepper's eyes turn **blue** while it is listening and **red** while it
  thinks or speaks.
* Face the robot when you talk to it. A new face is enrolled automatically
  the first time that person speaks to the robot.
* Try: introducing yourself and your hobbies; describing a friend ("My friend
  Alex loves climbing"); asking "Do you remember me?"; "What am I holding?";
  "Can you give me a high five?"; "Can you dance for me?"; "Please be silent."

## Running without Docker

**Server** (Python 3.10, CUDA 11.8, `ffmpeg` on the `PATH`, and Neo4j ≥ 5.11
for vector indexes):

```bash
pip install --extra-index-url https://download.pytorch.org/whl/cu118 \
    torch==2.2.1+cu118 torchvision==0.17.1+cu118 torchaudio==2.2.1+cu118
pip install -r server/requirements.txt

export OPENAI_API_KEY=... NEO4J_PASSWORD=... NEO4J_URI=bolt://localhost:7687
export FACE_DB_DIR=$PWD/database/face_db DISPLAY_IMGS_DIR=$PWD/display_imgs
export PYTHONPATH=$PWD:$PWD/grpc_communication
cd server && python main.py
```

**Client** (Python 3.10):

```bash
pip install -r pepper_client/requirements.txt
# ARM64 (Raspberry Pi):
pip install pepper_client/qi-3.1.4-cp310-cp310-manylinux_2_35_aarch64.whl
# x86_64:
pip install qi==3.1.5

python pepper_client/client.py --ip <pepper-ip> --server <server-ip>:50051 \
    --listen tcp://<client-ip>:9559
```

Client options:

* `--mic-threshold`: front-mic energy that counts as speech (default 370).
  Raise it in noisy rooms.
* `--silence-loops`: quiet audio buffers that end an utterance (default 15).
* `--volume`: speaker volume, 0–100 (default 40).

## Configuration

| Variable | Used by | Default | Description |
|---|---|---|---|
| `OPENAI_API_KEY` | server | — | Required. LLM, VLM and embeddings |
| `GROK_API_KEY` | server | empty | Optional fallback LLM/VLM |
| `NEO4J_URI` | server | `bolt://localhost:7687` | Neo4j Bolt URI |
| `NEO4J_USER` / `NEO4J_PASSWORD` | server | `neo4j` / — | Neo4j credentials |
| `FACE_DB_DIR` | server | `/workspace/database/face_db` | Enrolled face embeddings |
| `DISPLAY_IMGS_DIR` | server | `/workspace/display_imgs` | Frames shown by the viewer on `:8003` |
| `PEPPER_IP` / `PEPPER_PORT` | client | `192.168.0.52` / `9559` | Robot address |
| `ARIS_SERVER` | client | `localhost:50051` | Server gRPC address |
| `CLIENT_LISTEN_URL` | client | `tcp://0.0.0.0:9559` | qi endpoint Pepper streams audio to |
| `PEPPER_VOLUME` | client | `40` | Speaker volume (0–100) |

**Language models.** This release calls OpenAI first: GPT-4o for the
reasoner, vision and movement APIs, GPT-4-turbo for speech, and
`text-embedding-3-small` for embeddings. xAI Grok (`grok-3` for speech,
`grok-2` for the reasoner, `grok-2-vision` for vision) is used when an OpenAI
call fails. The user study in the paper ran with Grok 2 / Grok-2-Vision as the
primary models. Change the model calls in `server/reasoner/reasoner.py`,
`server/apis/speaking.py` and `server/apis/person_attr.py` to reproduce that
setup.

**Robot persona.** The prompts call the robot "Ginny" (the name used in the
user study). The reasoner also accepts common ASR variants (Jenny, Genie, …).
To rename it, edit the prompts in `server/reasoner/prompt.py`,
`server/apis/speaking.py`, `server/apis/movement/*_prompt.py` and
`server/core_api/*/prompt.py`.

**Dances.** Standard movements play animations from a Choregraphe package
named `hapy_dance` installed on the robot. Without it, the client falls back to
Pepper's built-in `happy` animations.

## Extending ARIS with a new API

1. Create a class that subclasses `ApiBase` in `server/apis/`. Implement
   `__call__(self, person_details)` as a generator that yields
   `ApiObject(text, mode=...)`. Plain text is spoken by the client. For a
   robot action, yield JSON and give it a mode the client handles.
2. Register it under a new state name in `api_call` in
   `server/apis/__init__.py`.
3. Teach the Reasoner when to choose that state by adding a rule and examples
   to `server/reasoner/prompt.py`.
4. If the API returns a new `mode`, handle it in
   `SpeechProcessor.execute_response` in `pepper_client/client.py`.

## Regenerating gRPC stubs

After editing `grpc_communication/grpc.proto`:

```bash
pip install grpcio-tools==1.41.1 protobuf==3.20.3
bash grpc_communication/regen.sh
```

## Privacy

For every person it meets, the server stores:

* a face embedding and the full camera frame used to enrol them
  (`database/face_db/face_<n>.npy` and `.png`),
* their complete conversation history, attributes and relationships (Neo4j),
* the most recent processed camera frame (`display_imgs/`).

Audio is written to a temporary file only for transcription and deleted right
after. Get informed consent before deploying. To erase all stored data, delete
the `database/` and `display_imgs/` directories and the `neo4j-data` volume
(`docker compose down -v`).

## Third-party components

| Component | Model / tool | License / terms |
|---|---|---|
| Speech recognition | Whisper large-v3 | MIT |
| Reasoner / speech / vision LLM | OpenAI GPT-4o / GPT-4-turbo; xAI Grok | Provider terms of service |
| Embeddings | OpenAI `text-embedding-3-small` | OpenAI Services Agreement |
| Face recognition | InsightFace `buffalo_l` | Non-commercial research use |
| Person detection | Ultralytics YOLOv8 | AGPL-3.0 |
| Graph / vector database | Neo4j Community 5.25 | GPLv3 |
| Robot SDK | libqi (Aldebaran / SoftBank Robotics) | BSD-3-Clause |
