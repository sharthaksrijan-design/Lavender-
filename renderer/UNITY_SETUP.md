# LAVENDER — Unity Renderer Setup
## Milestone 3: Hologram Display

---

## Quick Start

### 1. Install Unity

- Download **Unity Hub**: https://unity.com/download
- Install **Unity 2022.3 LTS** (exact version matters for package compatibility)
- When installing, include: **Universal Render Pipeline** support

### 2. Create the Project

```
Unity Hub → New Project → 3D (URP) → Name: "LavenderRenderer"
Location: /opt/lavender/renderer/
```

### 3. Install Required Packages

Open **Window → Package Manager**. Install these:

**From Package Manager (Unity Registry):**
- TextMeshPro (should already be included)
- Universal RP

**From git URL** (click "+" → "Add package from git URL"):
```
https://github.com/endel/NativeWebSocket.git#upm
```

**From NuGet (for Newtonsoft.Json):**
- Install NuGetForUnity first: https://github.com/GlitchEnzo/NuGetForUnity
- Then install: `Newtonsoft.Json`

### 4. Copy Scripts

Copy everything from `renderer/Assets/Scripts/` into your Unity project's `Assets/Scripts/` folder.

### 5. Scene Setup

Create this hierarchy in your scene:

```
Scene
├── LavenderSystem (empty GameObject)
│   └── LavenderDirectorClient.cs  ← attach here
│
├── Canvas (Screen Space - Camera, render to Display 2)
│   ├── Background (Image)           ← set in ThemeManager
│   ├── MasterCanvasGroup             ← CanvasGroup component
│   │
│   ├── Layer_Background (empty RectTransform, full screen)
│   │   └── AmbientDisplay.cs  ← attach here
│   │       ├── ClockText (TMP)
│   │       ├── DateText (TMP)
│   │       ├── EpochText (TMP)
│   │       └── StatsPanel
│   │           ├── CPURow (TMP + Slider)
│   │           ├── RAMRow (TMP + Slider)
│   │           └── GPURow (TMP + Slider)
│   │
│   ├── Layer_Persistent (empty RectTransform, full screen)
│   ├── Layer_Active (empty RectTransform, full screen)
│   ├── Layer_Foreground (empty RectTransform, full screen)
│   │
│   └── WaveformContainer (bottom strip, ~1920x80)
│       └── WaveformRenderer.cs  ← attach here (+ LineRenderer)
│
└── ThemeManager (empty GameObject)
    └── ThemeManager.cs  ← attach here
```

### 6. Configure ThemeManager

In ThemeManager Inspector:
- **Master Canvas Group** → drag the Canvas's CanvasGroup
- **Background Panel** → drag the Background Image

### 7. Configure LavenderDirectorClient

In Inspector:
- **Server Url**: `ws://localhost:8765`
- **Reconnect Delay**: `3`
- **Theme Manager** → drag ThemeManager
- **Panel Manager** → drag PanelManager (create this GameObject)
- **Waveform Renderer** → drag WaveformContainer
- **Ambient Display** → drag Layer_Background

### 8. Configure Display Output

- Go to **Edit → Project Settings → Player**
- Set **Display** count to 2
- In Canvas settings: set **Target Display** to **Display 2**
- Run the app — it will render to the projector on Display 2

### 9. Panel Prefab Setup

Create a `Panel` prefab in `Assets/Prefabs/`:

```
Panel (GameObject)
├── Image (background, set alpha to 0.06)
├── TopEdge (Image, 1px tall, anchored to top)
├── Title (TMP_Text)
└── Content (empty RectTransform with Vertical Layout Group)
```

Assign this prefab to PanelManager's `Panel Prefab` field.

---

## Testing Without Hardware

You can test the renderer on a single monitor:
1. Run `python core/hologram.py` — starts the WebSocket server
2. Press Play in Unity
3. The renderer will connect and start receiving directives
4. Open the hologram.py test loop — it will cycle personalities every 5s

---

## Display Calibration for Projector

When projecting onto the holographic film:

1. Set projector resolution to match film dimensions
2. In Unity: **Edit → Project Settings → Player → Resolution** → match exactly
3. Use a **mask shader** to crop out any frame/border artifacts
4. Adjust `ambientBrightness` in UIConfig.cs to match your room lighting
5. For Pepper's Ghost setups: flip the image horizontally in Player Settings

---

## Required Fonts

Install these fonts in Unity (download from Google Fonts):
- **Space Grotesk** → for all UI text
- **JetBrains Mono** → for data/code displays

Import each as a TMP Font Asset: `Window → TextMeshPro → Font Asset Creator`

---

## Performance Notes

- Target: **60fps** on the rendering machine
- The renderer is lightweight — all logic runs in Python
- If frame rate drops: reduce WaveformSamples in UIConfig.cs (128 → 64)
- Disable `logDirectives` in LavenderDirectorClient Inspector for production
