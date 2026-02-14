# Ginny Conversational Robot

This repository contains the software powering **Ginny**, a Pepper-based conversational robot.  The system is split into a server component that runs perception, reasoning and gRPC services, and a client component that runs directly on the Pepper robot.

## Repository Layout

```
 ginny_server/          – gRPC server and AI/vision logic
 pepper_client/        – Pepper-side middleware and utilities
 pepper_auto/          – gRPC service to execute robot motions
 grpc_communication/   – Protobuf definitions and generated stubs
 statistical_analysis/ – Jupyter notebook for data analysis
 test/                 – Example scripts for gRPC communication
```

### Server
* `ginny_server/main.py` starts the gRPC server on port **50051**.  The `MediaManager` servicer processes audio and images from the robot, transcribes speech with Whisper, detects faces, calls an LLM-based `Reasoner`, and streams back responses.
* Decisions from the `Reasoner` set a *state* which an `Executor` maps to APIs in `ginny_server/apis/`.  These APIs cover speech, movement, vision, and interactions with the Pepper robot through the secondary channel.
* Conversation history and face details are stored in a Neo4j database via utilities in `ginny_server/utils/neo4j_db/`.

Server dependencies are listed in `server_requirements.txt`.

### Client
* `pepper_client/pepper_middleware/pepper.py` connects to the Pepper robot using the Qi SDK, captures audio/video, streams data to the server and plays back responses.
* `pepper_client/environment.yml` defines a Python 2.7 Conda environment suitable for Pepper's SDK.
* Additional modules under `pepper_client/` implement camera/audio helpers, movement controllers, and a secondary channel for tasks such as object finding.

### Communication Protocol
Protobuf definitions live in `grpc_communication/`:
* `grpc.proto` defines the main `MediaService` for streaming audio/image data and a `SecondaryChannel` for auxiliary tasks.
* `pepper_auto.proto` describes the `PepperAuto` service used for direct robot control (fetching an image, executing joint motions, or making Pepper speak).

Generated Python stubs are committed alongside the `.proto` files.

### Testing Scripts
The `test/sending_audio` directory provides examples that send audio (and optional images) to the server and print the streamed LLM responses.

## Getting Started

1. **Install server dependencies** (Python 3.10 recommended):
   ```bash
   pip install -r server_requirements.txt
   ```
2. **Run the gRPC server**:
   ```bash
   python ginny_server/main.py
   ```
3. **Set up the Pepper client environment** on the robot using `pepper_client/environment.yml` and run the middleware:
   ```bash
   python pepper_client/pepper_middleware/pepper.py --qi-url <PEPPER_URL>
   ```
4. **Experiment** with the scripts under `test/` to send audio or images directly from a desktop machine.

## Next Steps
* Explore the reasoning pipeline in `ginny_server/reasoner` and how it selects APIs in `ginny_server/apis`.
* Review vision and LLM modules in `ginny_server/core_api` for face recognition, YOLO person detection, or language model integration.
* Extend `pepper_client` utilities to add new robot behaviors or secondary tasks.

