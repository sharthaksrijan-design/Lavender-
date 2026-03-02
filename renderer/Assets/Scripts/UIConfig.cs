// LAVENDER — UI Configuration
// renderer/Assets/Scripts/UIConfig.cs
//
// Single source of truth for all colors, timing, and layout constants.
// Every other script reads from here. Nothing is hardcoded elsewhere.
//
// To change a personality's look: edit its PersonalityTheme entry below.

using UnityEngine;
using System.Collections.Generic;

namespace Lavender
{
    [System.Serializable]
    public class PersonalityTheme
    {
        public string name;

        // Primary text and UI color
        public Color primaryColor;

        // Accent — buttons, highlights, active elements
        public Color accentColor;

        // Panel background (very low alpha — 0.05 to 0.10)
        public Color panelBackground;

        // Background canvas color
        public Color backgroundColor;

        // Waveform color
        public Color waveformColor;

        // Ambient idle waveform amplitude multiplier
        public float idleWaveformScale;

        // Panel animation duration (seconds)
        public float animationDuration;

        // How panels enter: "fade" | "slide" | "materialize" | "instant"
        public string panelEnterStyle;

        // Ambient brightness multiplier (0.5 = dim, 1.0 = full)
        public float ambientBrightness;

        // How much text opacity reduces for unfocused elements (0 = invisible, 1 = full)
        public float unfocusedTextOpacity;

        // Clock format: "HH:mm" = 24hr, "hh:mm tt" = 12hr
        public string clockFormat;

        // Whether to show epoch time alongside normal time (Vector only)
        public bool showEpochTime;

        // Waveform style: "sine" | "bars" | "particles" | "line"
        public string waveformStyle;
    }

    public static class UIConfig
    {
        // ── PERSONALITY THEMES ──────────────────────────────────────────────

        public static readonly Dictionary<string, PersonalityTheme> Themes
            = new Dictionary<string, PersonalityTheme>
        {
            ["iris"] = new PersonalityTheme
            {
                name                 = "iris",
                primaryColor         = new Color(0.91f, 0.96f, 1.00f, 1.0f),  // #E8F4FF cold white
                accentColor          = new Color(0.31f, 0.76f, 0.97f, 1.0f),  // #4FC3F7 ice blue
                panelBackground      = new Color(1.00f, 1.00f, 1.00f, 0.05f),
                backgroundColor      = new Color(0.03f, 0.05f, 0.06f, 1.0f),  // #080C10
                waveformColor        = new Color(0.31f, 0.76f, 0.97f, 0.6f),
                idleWaveformScale    = 0.15f,  // barely visible
                animationDuration    = 0.0f,   // instant — no transitions
                panelEnterStyle      = "instant",
                ambientBrightness    = 0.85f,
                unfocusedTextOpacity = 0.35f,
                clockFormat          = "HH:mm",
                showEpochTime        = false,
                waveformStyle        = "line",
            },

            ["nova"] = new PersonalityTheme
            {
                name                 = "nova",
                primaryColor         = new Color(1.00f, 0.97f, 0.94f, 1.0f),  // #FFF8F0 warm white
                accentColor          = new Color(1.00f, 0.70f, 0.28f, 1.0f),  // #FFB347 amber
                panelBackground      = new Color(1.00f, 0.97f, 0.94f, 0.07f),
                backgroundColor      = new Color(0.05f, 0.04f, 0.03f, 1.0f),  // #0D0A08
                waveformColor        = new Color(1.00f, 0.70f, 0.28f, 0.75f),
                idleWaveformScale    = 0.4f,
                animationDuration    = 0.25f,
                panelEnterStyle      = "slide",
                ambientBrightness    = 1.0f,
                unfocusedTextOpacity = 0.50f,
                clockFormat          = "hh:mm tt",
                showEpochTime        = false,
                waveformStyle        = "sine",
            },

            ["vector"] = new PersonalityTheme
            {
                name                 = "vector",
                primaryColor         = new Color(1.00f, 1.00f, 1.00f, 1.0f),  // pure white
                accentColor          = new Color(0.00f, 1.00f, 0.53f, 1.0f),  // #00FF88 electric green
                panelBackground      = new Color(1.00f, 1.00f, 1.00f, 0.06f),
                backgroundColor      = new Color(0.02f, 0.03f, 0.06f, 1.0f),  // #05080F deep navy
                waveformColor        = new Color(0.00f, 1.00f, 0.53f, 0.8f),
                idleWaveformScale    = 0.0f,   // replaced by stats strip
                animationDuration    = 0.10f,  // fast
                panelEnterStyle      = "fade",
                ambientBrightness    = 1.0f,
                unfocusedTextOpacity = 0.45f,
                clockFormat          = "HH:mm",
                showEpochTime        = true,
                waveformStyle        = "bars",
            },

            ["solace"] = new PersonalityTheme
            {
                name                 = "solace",
                primaryColor         = new Color(0.96f, 0.94f, 0.92f, 1.0f),  // #F5F0EB warm grey
                accentColor          = new Color(0.79f, 0.63f, 0.63f, 1.0f),  // #C9A0A0 dusty rose
                panelBackground      = new Color(0.96f, 0.94f, 0.92f, 0.05f),
                backgroundColor      = new Color(0.03f, 0.03f, 0.02f, 1.0f),  // #080706
                waveformColor        = new Color(0.79f, 0.63f, 0.63f, 0.5f),
                idleWaveformScale    = 0.0f,   // replaced by particle drift
                animationDuration    = 0.55f,  // very slow
                panelEnterStyle      = "fade",
                ambientBrightness    = 0.7f,
                unfocusedTextOpacity = 0.40f,
                clockFormat          = "HH:mm",
                showEpochTime        = false,
                waveformStyle        = "particles",
            },

            ["lilac"] = new PersonalityTheme
            {
                name                 = "lilac",
                primaryColor         = new Color(0.94f, 0.92f, 0.97f, 1.0f),  // #F0EAF8 cold white
                accentColor          = new Color(0.61f, 0.35f, 0.71f, 1.0f),  // #9B59B6 sharp violet
                panelBackground      = new Color(0.61f, 0.35f, 0.71f, 0.06f),
                backgroundColor      = new Color(0.10f, 0.04f, 0.18f, 1.0f),  // #1A0A2E deep violet
                waveformColor        = new Color(0.61f, 0.35f, 0.71f, 0.85f),
                idleWaveformScale    = 0.2f,
                animationDuration    = 0.0f,   // materialize — no smooth transitions
                panelEnterStyle      = "materialize",
                ambientBrightness    = 0.9f,
                unfocusedTextOpacity = 0.40f,
                clockFormat          = "HH:mm",
                showEpochTime        = false,
                waveformStyle        = "line",
            },
        };

        // ── TYPOGRAPHY ──────────────────────────────────────────────────────

        // Font size in Unity units (1080p canvas, scale to match)
        public const float FontSizeDisplay  = 96f;
        public const float FontSizeHeadline = 36f;
        public const float FontSizeBody     = 20f;
        public const float FontSizeCaption  = 13f;
        public const float FontSizeData     = 18f;

        // ── LAYOUT ──────────────────────────────────────────────────────────

        // Canvas reference resolution
        public const float CanvasWidth  = 1920f;
        public const float CanvasHeight = 1080f;

        // Panel corner radius
        public const float PanelCornerRadius = 8f;

        // Panel top-edge opacity (the only border panels have)
        public const float PanelEdgeOpacity = 0.20f;

        // Panel internal padding
        public const float PanelPaddingX = 24f;
        public const float PanelPaddingY = 16f;

        // Panel layer Z offsets (lower = closer to viewer)
        public const float LayerZBackground = 400f;
        public const float LayerZPersistent = 300f;
        public const float LayerZActive     = 200f;
        public const float LayerZForeground = 100f;

        // Default panel width as fraction of canvas width
        public const float DefaultPanelWidthFraction = 0.38f;

        // ── WAVEFORM ────────────────────────────────────────────────────────

        public const int   WaveformSamples       = 128;
        public const float WaveformHeight        = 60f;
        public const float WaveformUpdateHz      = 30f;
        public const float WaveformIdleFrequency = 0.4f;   // Hz — slow sine
        public const float WaveformSpeakSmoothing = 0.15f;

        // ── TRANSITION ──────────────────────────────────────────────────────

        // Personality switch — collapse inward then expand outward
        public const float TransitionCollapseDuration = 0.6f;
        public const float TransitionHoldDuration     = 0.3f;
        public const float TransitionExpandDuration   = 0.9f;
        public const float TransitionTotalDuration    = 1.8f;

        // ── ALERT OVERRIDE ──────────────────────────────────────────────────

        public static readonly Color AlertInfoColor     = new Color(0.31f, 0.76f, 0.97f);
        public static readonly Color AlertWarningColor  = new Color(1.00f, 0.70f, 0.28f);
        public static readonly Color AlertCriticalColor = new Color(1.00f, 0.27f, 0.27f);
        public const float AlertPulseDuration = 0.3f;

        // ── HELPERS ─────────────────────────────────────────────────────────

        public static PersonalityTheme GetTheme(string personality)
        {
            if (Themes.TryGetValue(personality.ToLower(), out var theme))
                return theme;

            Debug.LogWarning($"[UIConfig] Unknown personality '{personality}', using nova.");
            return Themes["nova"];
        }

        public static Color WithAlpha(Color c, float a)
            => new Color(c.r, c.g, c.b, a);
    }
}
