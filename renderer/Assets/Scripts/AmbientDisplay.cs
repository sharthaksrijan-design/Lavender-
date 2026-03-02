// LAVENDER — Ambient Display
// renderer/Assets/Scripts/AmbientDisplay.cs
//
// Manages the always-on ambient information layer:
//   - Clock (time + date, format depends on personality)
//   - System stats (CPU / RAM / GPU bars)
//   - Epoch time (Vector mode only)
//   - Idle state indicator
//
// This is Layer 0 — always present when someone is at the table,
// never fully cleared, just dimmed by the brightness system.
//
// Attach to: an empty GameObject called "AmbientDisplay" in the scene.

using System;
using System.Collections;
using UnityEngine;
using TMPro;
using UnityEngine.UI;

namespace Lavender
{
    public class AmbientDisplay : MonoBehaviour
    {
        // ── SINGLETON ────────────────────────────────────────────────────────
        public static AmbientDisplay Instance { get; private set; }

        // ── INSPECTOR ────────────────────────────────────────────────────────

        [Header("Clock")]
        [SerializeField] private TMP_Text timeText;
        [SerializeField] private TMP_Text dateText;
        [SerializeField] private TMP_Text epochText;   // Vector mode only
        [SerializeField] private GameObject epochRow;

        [Header("System Stats")]
        [SerializeField] private TMP_Text cpuLabel;
        [SerializeField] private TMP_Text ramLabel;
        [SerializeField] private TMP_Text gpuLabel;
        [SerializeField] private Slider   cpuBar;
        [SerializeField] private Slider   ramBar;
        [SerializeField] private Slider   gpuBar;
        [SerializeField] private GameObject statsPanel;

        [Header("State Indicator")]
        [SerializeField] private Image  stateIndicatorDot;
        [SerializeField] private TMP_Text stateLabel;

        [Header("Update intervals")]
        [SerializeField] private float clockUpdateInterval = 1f;
        [SerializeField] private float statsUpdateInterval = 5f;

        // ── INTERNALS ─────────────────────────────────────────────────────────

        private Coroutine _clockCoroutine;
        private Coroutine _statsPulseCoroutine;

        private float _cpuPct = 0f;
        private float _ramPct = 0f;
        private float _gpuPct = 0f;

        private string _currentState = "ambient";

        // State indicator colors (match Python SystemState)
        private static readonly Color StateColorAmbient  = new Color(0.4f, 0.4f, 0.4f);
        private static readonly Color StateColorActive   = new Color(0.3f, 0.9f, 0.5f);
        private static readonly Color StateColorFocus    = new Color(0.3f, 0.6f, 1.0f);
        private static readonly Color StateColorThinking = new Color(1.0f, 0.7f, 0.2f);
        private static readonly Color StateColorSpatial  = new Color(0.7f, 0.3f, 1.0f);

        // ── UNITY ────────────────────────────────────────────────────────────

        private void Awake()
        {
            if (Instance != null && Instance != this)
            {
                Destroy(gameObject);
                return;
            }
            Instance = this;
        }

        private void OnEnable()
        {
            ThemeManager.OnThemeChanged += OnThemeChanged;
            _clockCoroutine = StartCoroutine(ClockLoop());
        }

        private void OnDisable()
        {
            ThemeManager.OnThemeChanged -= OnThemeChanged;
            if (_clockCoroutine != null) StopCoroutine(_clockCoroutine);
        }

        // ── PUBLIC API — called by LavenderDirectorClient ─────────────────

        /// <summary>
        /// Update clock display from Python-pushed time strings.
        /// Python pushes these every 5 seconds so they stay in sync.
        /// </summary>
        public void UpdateClock(string timeStr, string dateStr)
        {
            if (timeText != null)
            {
                timeText.text = timeStr;
                ThemeManager.StyleText(timeText, true);
            }
            if (dateText != null)
            {
                dateText.text = dateStr;
                ThemeManager.StyleText(dateText, false);
            }

            // Epoch time (Vector mode)
            if (epochText != null)
            {
                long epoch = DateTimeOffset.UtcNow.ToUnixTimeSeconds();
                epochText.text = $"epoch {epoch}";
                ThemeManager.StyleText(epochText, false);
            }
        }

        /// <summary>
        /// Update system stats from Python-pushed data.
        /// </summary>
        public void UpdateStats(float cpuPct, float ramPct, float gpuPct)
        {
            _cpuPct = cpuPct;
            _ramPct = ramPct;
            _gpuPct = gpuPct;

            if (cpuLabel != null) cpuLabel.text = $"CPU  {cpuPct:F0}%";
            if (ramLabel != null) ramLabel.text = $"RAM  {ramPct:F0}%";
            if (gpuLabel != null) gpuLabel.text = $"GPU  {gpuPct:F0}%";

            SetBarValue(cpuBar, cpuPct / 100f);
            SetBarValue(ramBar, ramPct / 100f);
            SetBarValue(gpuBar, gpuPct / 100f);

            // Color bars by threshold
            ColorBar(cpuBar, cpuPct);
            ColorBar(ramBar, ramPct);
            ColorBar(gpuBar, gpuPct);
        }

        /// <summary>
        /// Update the state indicator based on current system state.
        /// </summary>
        public void SetState(string state)
        {
            _currentState = state;

            Color dotColor = state switch
            {
                "active"   => StateColorActive,
                "focus"    => StateColorFocus,
                "thinking" => StateColorThinking,
                "spatial"  => StateColorSpatial,
                _          => StateColorAmbient,
            };

            if (stateIndicatorDot != null)
            {
                // Pulse on state change
                if (_statsPulseCoroutine != null) StopCoroutine(_statsPulseCoroutine);
                _statsPulseCoroutine = StartCoroutine(PulseDot(dotColor));
            }

            if (stateLabel != null)
            {
                stateLabel.text = state.ToUpper();
                ThemeManager.StyleText(stateLabel, false);
            }
        }

        // ── CLOCK LOOP ─────────────────────────────────────────────────────

        // Local clock loop runs independently of Python pushes
        // as a fallback so display is never stale
        private IEnumerator ClockLoop()
        {
            while (true)
            {
                UpdateLocalClock();
                yield return new WaitForSeconds(clockUpdateInterval);
            }
        }

        private void UpdateLocalClock()
        {
            var theme = ThemeManager.Current;
            if (theme == null) return;

            string fmt = theme.clockFormat ?? "HH:mm";
            var now = DateTime.Now;

            if (timeText != null)
            {
                timeText.text = now.ToString(fmt);
                ThemeManager.StyleText(timeText, true);
            }

            if (dateText != null)
            {
                // Only update date — Python overrides this with a formatted string
                dateText.text = now.ToString("dddd");
                ThemeManager.StyleText(dateText, false);
            }

            // Epoch time — Vector only
            bool showEpoch = theme.showEpochTime;
            if (epochRow != null) epochRow.SetActive(showEpoch);
            if (showEpoch && epochText != null)
            {
                long epoch = DateTimeOffset.UtcNow.ToUnixTimeSeconds();
                epochText.text = $"epoch {epoch}";
                ThemeManager.StyleText(epochText, false);
            }
        }

        // ── STAT BARS ────────────────────────────────────────────────────────

        private void SetBarValue(Slider bar, float value)
        {
            if (bar != null)
            {
                // Animate to new value
                StopAllCoroutines();
                StartCoroutine(AnimateBar(bar, bar.value, value, 0.4f));
            }
        }

        private IEnumerator AnimateBar(Slider bar, float from, float to, float duration)
        {
            float elapsed = 0f;
            while (elapsed < duration)
            {
                bar.value = Mathf.Lerp(from, to, elapsed / duration);
                elapsed += Time.deltaTime;
                yield return null;
            }
            bar.value = to;
        }

        private void ColorBar(Slider bar, float pct)
        {
            if (bar == null) return;
            var fill = bar.fillRect?.GetComponent<Image>();
            if (fill == null) return;

            fill.color = pct > 85f
                ? new Color(1f, 0.27f, 0.27f)
                : pct > 65f
                    ? new Color(1f, 0.70f, 0.28f)
                    : ThemeManager.Current?.accentColor ?? Color.green;
        }

        // ── STATE PULSE ───────────────────────────────────────────────────────

        private IEnumerator PulseDot(Color targetColor)
        {
            if (stateIndicatorDot == null) yield break;

            // Flash to white then settle into target color
            float duration = 0.3f;
            float elapsed  = 0f;

            stateIndicatorDot.color = Color.white;
            while (elapsed < duration)
            {
                stateIndicatorDot.color = Color.Lerp(Color.white, targetColor, elapsed / duration);
                elapsed += Time.deltaTime;
                yield return null;
            }
            stateIndicatorDot.color = targetColor;
        }

        // ── FOCUS MODE ────────────────────────────────────────────────────────

        /// <summary>
        /// Focus mode — hide stats, keep only clock.
        /// </summary>
        public void SetFocusMode(bool focused)
        {
            if (statsPanel != null) statsPanel.SetActive(!focused);
            if (dateText != null)   dateText.gameObject.SetActive(!focused);

            if (timeText != null)
            {
                // Enlarge time in focus mode
                timeText.fontSize = focused ? UIConfig.FontSizeDisplay * 1.2f : UIConfig.FontSizeDisplay;
                ThemeManager.StyleText(timeText, true);
            }
        }

        // ── THEME ─────────────────────────────────────────────────────────────

        private void OnThemeChanged(PersonalityTheme theme)
        {
            // Reapply text colors
            ThemeManager.StyleText(timeText, true);
            ThemeManager.StyleText(dateText, false);
            ThemeManager.StyleText(epochText, false);
            ThemeManager.StyleText(cpuLabel, false);
            ThemeManager.StyleText(ramLabel, false);
            ThemeManager.StyleText(gpuLabel, false);
            ThemeManager.StyleText(stateLabel, false);

            // Recolor bars
            ColorBar(cpuBar, _cpuPct);
            ColorBar(ramBar, _ramPct);
            ColorBar(gpuBar, _gpuPct);

            // Show/hide epoch row
            if (epochRow != null) epochRow.SetActive(theme.showEpochTime);
        }
    }
}
