<p align="center">
  <img src="assets/banner.svg" alt="dribble3d: Sketch it. Get a 3D model you can print." width="100%">
</p>

<p align="center">
  <img src="https://img.shields.io/badge/closed--loop-verified-2A9D8F?style=for-the-badge" alt="closed-loop">
  <img src="https://img.shields.io/badge/vision%20AI-sees%20your%20sketch-8E44AD?style=for-the-badge" alt="vision AI">
  <img src="https://img.shields.io/badge/multi--view-Zero123%2B%2B%20%E2%86%92%20Hunyuan3D-2980B9?style=for-the-badge" alt="multi-view">
  <img src="https://img.shields.io/badge/TRELLIS.2-textured-E76F51?style=for-the-badge" alt="TRELLIS.2">
  <img src="https://img.shields.io/badge/deploy-Modal-1F2937?style=for-the-badge" alt="Modal">
</p>

<h3 align="center">Draw something rough. Get a 3D model you can print.<br>And watch every step of how it got there.</h3>

<p align="center">
  <img src="assets/demo.gif" alt="dribble3d demo" width="85%">
</p>

---

<h1 align="center">🌐 Try it</h1>

<p align="center"><b><a href="https://crazycat8686--dribble3d-dribble-web.us-east.modal.run/">crazycat8686--dribble3d-dribble-web.us-east.modal.run</a></b></p>

> [!NOTE]
> **Deployed and working.** The app runs on Modal with an L40S GPU. It sleeps when nobody is using it, so the first visit after a quiet spell takes **1 to 2 minutes** to wake up. Give it a moment.

---

<h1 align="center">💡 The idea</h1>

Turning a drawing into a 3D object is not one problem. It is five or six, and no single AI model is good at all of them. So dribble3d does not lean on one model. It brings in the **right specialist for each job**, lets a **vision AI look at your drawing** to understand what it is, and then **checks its own work** against what you drew. If something is off, you can see it, and you can fix it.

<table>
<tr>
<td width="33%" valign="top">

### 🧩 The right model for each job
A model that understands pictures, one that paints realistic images, one that cuts objects out, and different ones that build 3D. Each is used because the step needs it.

</td>
<td width="33%" valign="top">

### 👁️ An AI that actually looks
A vision-language model reads your sketch and names what it sees. A second model, CLIP, double-checks that name. You see both opinions, not just the verdict.

</td>
<td width="33%" valign="top">

### 🔁 A loop that corrects itself
The result is compared back to your drawing. Not a match? It tries again with a better candidate. You can also step in and steer it.

</td>
</tr>
</table>

---

<h1 align="center">🧭 How it works</h1>

<p align="center">
  <img src="assets/flow.svg" alt="dribble3d pipeline" width="100%">
</p>

1. **Read the drawing.** Your strokes and outline are extracted.
2. **Understand it.** A vision AI looks at the sketch and says what it is. CLIP weighs in as a second opinion. If they disagree, CLIP picks the name that fits the drawing better.
3. **Imagine it.** SDXL with a ControlNet turns your sketch into several realistic pictures of the object.
4. **Choose the best one.** Each picture is scored on how closely it follows your lines, how well its shape matches, how well it fits the label, and how easy it will be to turn into 3D.
5. **Build it.** A 3D model is made from the winning picture.
6. **Check it.** The model is flattened back to 2D and compared with your sketch. If the match is weak, we go back to step 5 with the next-best picture.

Every stage reports its own timing, and the result page shows all of it.

---

<h1 align="center">👁️ Seeing is the point</h1>

Most sketch-to-3D tools are a text box, a spinner and a mesh. If the result is wrong, you never find out why. We wanted the opposite: nothing hidden.

### You see what the AI thinks your drawing is
The **Identified as** card shows the label, which model produced it, whether CLIP agrees, and the alternatives with their confidence. Wrong guess? Tap another option, or type your own.

### You see every candidate, and its score
All the generated pictures sit side by side with the winner highlighted. The score is broken into its parts, so you can tell *why* one beat the others, and notice when none were good.

<p align="center">
  <img src="assets/scoring.svg" alt="Candidate scoring formula" width="100%">
</p>

### You see how well the 3D matches your drawing
After the model is built, we project it back to 2D and measure its overlap with your original sketch. That number, and whether a retry was needed, is on the result page.

### You see the whole machine
Timings per stage, which backend built the mesh, triangle count, size, volume, whether the file reloads cleanly, and whether it is print-ready. All in one **Under the hood** panel.

### You are told when something went sideways
If multi-view failed and we fell back to single view, or the background remover took a bite out of your object, you get a plain note. You would rather know than get a quietly worse model.

---

<h1 align="center">🔁 The correction loop</h1>

<p align="center">
  <img src="assets/loops.svg" alt="Two feedback loops" width="100%">
</p>

The loop runs on its own, and you can join it at any point.

**On its own**
- **Pick the best 2D picture.** 3 candidates in Fast mode, 5 in Quality. The runner-up is kept as a backup.
- **Check the 3D against your sketch.** Below an overlap of **0.60**, it retries with the backup and keeps whichever model matches better.
- **Multi-view tries more than once.** Zero123++ has to invent the sides it cannot see, so each run differs. We build up to **3 versions**, score each on sketch match and side and back consistency, stop early when one is good, and keep the best.
- **If everything fails,** it drops to a lighter option instead of crashing.

**With you**
- **Fix the label.** Pick a different guess or type your own, and the whole pipeline follows it.
- **Refine in plain words.** The **Refine** box understands things like "red brick walls, grey roof", "carved wooden" or "thinner", and turns them into prompt changes and control-strength changes.

---

<h1 align="center">🖼️ More than one picture, on purpose</h1>

<p align="center">
  <img src="assets/multi_image.svg" alt="Multiple images" width="100%">
</p>

One picture cannot show you the back of an object. So we use several, in three different ways:

| Where | What | Why |
|---|---|---|
| **Candidates** | Several 2D renders of the same sketch, scored and ranked | One unlucky sample does not ruin the result |
| **Multi-view** | Zero123++ turns the winner into 6 views, and Hunyuan3D-2mv builds the mesh from front, side and back | The back and sides stop being pure guesswork |
| **4 Photos** | Real front, left, back and right photos, each cut out and checked against the finished mesh from its own angle | Real photos beat invented ones |

---

<h1 align="center">🧱 Three ways to build the 3D</h1>

| | ⚡ **Single view** | 🔭 **Multi-view** *(default)* | 🎨 **TRELLIS.2** |
|---|---|---|---|
| **Model** | TripoSR | Zero123++ → Hunyuan3D-2mv | Microsoft TRELLIS.2-4B |
| **Best for** | Simple, compact objects | Objects where back and sides matter | Shape **and colour** in one pass |
| **Speed** | Fastest | Medium | Slowest |
| **GPU** | Standard | Standard | Large (own container) |

**Three ways in:** ✏️ **Draw** in the browser · 📤 **Upload** a drawing or photo (it works out which) · 📷 **4 Photos** (front plus at least one more angle).

When it is done, re-light the model with a material preset (Original, Clay, Terracotta, Metallic gold, Toy matte) and download it as **`.glb`**, **`.obj`** or **`.stl`**.

---

<h1 align="center">🧠 The models, and why each is there</h1>

| Job | Model | Why it is needed |
|---|---|---|
| **Understand the sketch** | Qwen2.5-VL-7B-Instruct, local, no API key | Reads the drawing and names it in plain words |
| **Second opinion** | CLIP ViT-L/14 | Cross-checks the name and scores how well each picture fits it |
| **Imagine the object** | SDXL + xinsir scribble ControlNet + fp16-fix VAE (SD 1.5 + lllyasviel scribble as a lighter fallback) | Turns rough lines into realistic pictures that still follow your shape |
| **Cut it out** | rembg `isnet-general-use` (u2net fallback) + a clean-up pass | Isolates the object and repairs edges and see-through gaps |
| **Fast 3D** | TripoSR | Quick answer for simple objects |
| **Better 3D** | Zero123++ v1.2 → Hunyuan3D-2mv | Invents the missing sides, then builds from several views |
| **3D with colour** | TRELLIS.2-4B (+ DINOv3, RMBG-2.0) | Shape and colour together |
| **Make it printable** | trimesh, pymeshlab, fast-simplification | Closes holes, smooths and trims the mesh |
| **Serve it** | FastAPI on Modal (L40S GPUs) | One file, one URL |

---

<h1 align="center">📸 Examples</h1>

> Videos and final renders coming soon.

### 🛋️ Sofa · Multi-view
The back, the arms and the seat depth all have to be inferred. A good test for multi-view.

| Sketch | Candidates + scores | 3D result |
|:---:|:---:|:---:|
| ![sofa sketch](assets/sofa_sketch.png) | ![sofa candidates](assets/sofa_candidates.png) | ![sofa 3d](assets/sofa_3d.png) |

<!-- video: assets/sofa.mp4 -->

### 🐔 Chicken · TRELLIS.2
Shape and colour come out together, so the feathers and comb keep their colours.

| Sketch | 3D result (TRELLIS.2) |
|:---:|:---:|
| ![chicken sketch](assets/chicken_sketch.png) | ![chicken 3d](assets/chicken_3d.png) |

<!-- video: assets/chicken.mp4 -->

### ✋ Hand · the stress test
Fingers hide each other, parts are thin, and every angle looks different. We include it on purpose: it shows the scoring, the retry loop, and the notes the app gives when the answer is not perfect.

| Input | Candidates + scores | 3D result |
|:---:|:---:|:---:|
| ![hand input](assets/hand_input.png) | ![hand candidates](assets/hand_candidates.png) | ![hand 3d](assets/hand_3d.png) |

<!-- video: assets/hand.mp4 -->

---

<h1 align="center">🚀 Run your own copy</h1>

### What you need

- A [Modal](https://modal.com) account (GPU time is billed per second, only while the app runs)
- Python 3.9 or newer, and `git`
- For TRELLIS.2: a Hugging Face account and access token

### 1. Get the code

```bash
git clone <your-repo-url> dribble3d
cd dribble3d
```

Keep `dribble3d.py` and `modal_app.py` in the same folder.

### 2. Set up Modal

```bash
pip install modal
modal setup
```

### 3. Allow TRELLIS.2 to download its models

1. On Hugging Face, accept the licences on both gated pages:
   `facebook/dinov3-vitl16-pretrain-lvd1689m` and `briaai/RMBG-2.0`
2. Create a read token under **Settings → Access Tokens**
3. Give it to Modal:

```bash
modal secret create huggingface-secret HF_TOKEN=hf_xxx
```

### 4. Deploy

```bash
modal deploy modal_app.py
```

> [!NOTE]
> The first deploy is slow. It builds the images, compiles several CUDA packages and bakes in tens of gigabytes of model weights. You only pay that cost once. Later deploys that change only code take seconds.

### 5. Open it

Modal prints a link when the deploy finishes. It looks like:

```
https://<your-modal-username>--dribble3d-dribble-web.<region>.modal.run
```

### 6. Make sure it is healthy

- Open `<your-link>/health`. You want `neural_ready` to be `true` and `errors` to be empty.
- Draw something simple, like a mug or a chair, and generate. You should see the label, the candidates with scores, and the 3D viewer.
- Download the `.glb`, `.obj` or `.stl`.

### Update or switch off

```bash
modal deploy modal_app.py      # after changing code
modal app stop dribble3d       # take it offline
```

### 📓 No Modal? Use the notebook

A notebook that runs the app without TRELLIS.2 is available on another branch of this repository. Open it there to try dribble3d on a GPU notebook environment.

### ❄️ Cold starts and cost

- The GPU container **sleeps when idle**, so you pay only while it works.
- The next visit reloads the models and takes **1 to 2 minutes**. The page shows waiting messages while it wakes.
- The first multi-view run after waking is slower, because Zero123++ and Hunyuan3D load on first use.
- The idle window is `scaledown_window` in `modal_app.py` (20 minutes).

---

<h1 align="center">⚙️ Settings</h1>

Set in the image `.env` of `modal_app.py`, or as environment variables locally.

| Variable | Default | What it does |
|---|---|---|
| `D3_SD` | `xl` | `xl` for SDXL, `15` for SD 1.5 |
| `D3_MV` / `D3_MV_DEFAULT` | `1` / `1` | Turn multi-view on / make it the default |
| `D3_MV_TRIES` | `3` | Most multi-view versions to build and compare |
| `D3_MV_GOOD` | `0.80` | Stop early when a try scores this well |
| `D3_IOU_RETRY` | `0.60` | Retry with the next candidate below this sketch match |
| `D3_HYMV_RES` | `448` | Hunyuan3D mesh detail |
| `D3_HYMV_STEPS` | `60` | Hunyuan3D diffusion steps |
| `D3_FACES` | `400000` | Triangle budget after simplifying |
| `D3_SMOOTH` | `3` | Smoothing passes |
| `D3_TRELLIS` | `1` | Turn the TRELLIS.2 option on |
| `D3_REMBG` | `isnet-general-use` | Background-removal model |

---

<h1 align="center">⚠️ Honest limits</h1>

> [!WARNING]
> We would rather tell you than have you find out.

- **One picture is not enough to know the back.** The loop and multi-view voting make it better, not solved.
- **Hands, thin legs and see-through objects** are hard. Try them, but do not expect perfect meshes.
- **Pale objects on white** (white walls, white mugs) can be partly erased by the background remover. When we notice, we restore the missing part from your outline and tell you.
- **The sketch match is a rough self-check, not the truth.** A loose or stylised sketch scores lower even when the model looks right. Judge by the model.
- **Cold starts are real** on a GPU that sleeps.
- **TRELLIS.2 is heavy.** It runs in its own container on a large GPU and is the slowest option.

---

<h1 align="center">📁 What is in the repo</h1>

```
.
├── dribble3d.py   # the whole app: pipeline, API and web page
├── modal_app.py   # Modal deployment (main app + separate TRELLIS.2 container)
├── assets/        # banner, diagrams, screenshots, GIFs and videos
└── README.md
```

---

<h3 align="center">🙏 Credits</h3>

<p align="center">
Built on TripoSR · Hunyuan3D-2 · TRELLIS.2 · Zero123++ · Stable Diffusion XL · ControlNet and xinsir scribble-SDXL · Qwen2.5-VL · CLIP · rembg · trimesh · pymeshlab.<br>
Check each model's licence before using the results commercially.
</p>

<h3 align="center">📄 License</h3>
<p align="center">Add your licence here.</p>
