// LAVENDER — Waveform Renderer
// renderer/Assets/Scripts/WaveformRenderer.cs
//
// Renders the audio waveform at the bottom of the display.
// Four states:
//   IDLE      — slow, low-amplitude sine. Barely visible. Lavender is present.
//   LISTENING — reacts to mic RMS level coming from Python.
//   THINKING  — slow, irregular, irregular pulse. Like deliberation.
//   SPEAKING  — live amplitude data pushed from voice output.
//
// Each personality changes: waveform color, style (sine/bars/particles/line),
// idle amplitude, and visibility.
//
// Attach to: a GameObject with a LineRenderer or RawImage component.
// Recommended: use LineRenderer for line/sine styles, custom mesh for bars.

using System;
using System.Collections;
using UnityEngine;
using UnityEngine.UI;

namespace Lavender
{
    [RequireComponent(typeof(LineRenderer))]
    public class WaveformRenderer : MonoBehaviour
    {
        // ── INSPECTOR ─────────────────────────────────────────────────────────

        [Header("References")]
        [SerializeField] private LineRenderer lineRenderer;
        [SerializeField] private RectTransform container;

        [Header("State")]
        [SerializeField] private string currentState = "idle";
        [SerializeField] private float  currentAmplitude = 0f;

        // ── INTERNALS ─────────────────────────────────────────────────────────

        private float[] _samples;
        private float   _time = 0f;
        private float   _targetAmplitude = 0f;
        private float   _smoothedAmplitude = 0f;

        // For thinking state — randomized wobble
        private float _thinkingPhase = 0f;
        private float _thinkingFreq  = 0.8f;

        // ── UNITY ────────────────────────────────────────────────────────────

        private void Awake()
        {
            _samples = new float[UIConfig.WaveformSamples];

            if (lineRenderer == null)
                lineRenderer = GetComponent<LineRenderer>();

            lineRenderer.positionCount = UIConfig.WaveformSamples;
            lineRenderer.useWorldSpace = false;
            lineRenderer.widthMultiplier = 2f;

            ApplyTheme(ThemeManager.Current);
        }

        private void OnEnable()
        {
            ThemeManager.OnThemeChanged += OnThemeChanged;
        }

        private void OnDisable()
        {
            ThemeManager.OnThemeChanged -= OnThemeChanged;
        }

        private void Update()
        {
            _time += Time.deltaTime;

            // Smooth amplitude transitions
            _smoothedAmplitude = Mathf.Lerp(
                _smoothedAmplitude,
                _targetAmplitude,
                UIConfig.WaveformSpeakSmoothing
            );

            UpdateSamples();
            RenderSamples();
        }

        // ── PUBLIC API ────────────────────────────────────────────────────────

        /// <summary>
        /// Set the waveform state. Called by LavenderDirectorClient.
        /// </summary>
        public void SetState(string state, float amplitude = 0f)
        {
            currentState = state.ToLower();
            _targetAmplitude = amplitude;

            // Thinking gets a random freq so it doesn't look mechanical
            if (currentState == "thinking")
                _thinkingFreq = UnityEngine.Random.Range(0.5f, 1.2f);
        }

        /// <summary>
        /// Push a live amplitude value during speaking.
        /// Called at ~30Hz from LavenderDirectorClient when audio is playing.
        /// </summary>
        public void PushAmplitude(float amplitude)
        {
            _targetAmplitude = amplitude;
        }

        // ── SAMPLE GENERATION ─────────────────────────────────────────────────

        private void UpdateSamples()
        {
            var theme = ThemeManager.Current;
            if (theme == null) return;

            switch (currentState)
            {
                case "idle":
                    GenerateIdle(theme);
                    break;

                case "listening":
                    GenerateListening();
                    break;

                case "thinking":
                    GenerateThinking();
                    break;

                case "speaking":
                    GenerateSpeaking();
                    break;

                default:
                    GenerateIdle(theme);
                    break;
            }
        }

        private void GenerateIdle(PersonalityTheme theme)
        {
            // Solace uses particles — handled separately; others get sine
            float scale = theme.idleWaveformScale;
            float freq  = UIConfig.WaveformIdleFrequency;

            for (int i = 0; i < _samples.Length; i++)
            {
                float t = (float)i / (_samples.Length - 1);
                // Gentle sine with slight second harmonic
                _samples[i] = scale * (
                    0.7f * Mathf.Sin(2f * Mathf.PI * freq * _time + t * 4f) +
                    0.3f * Mathf.Sin(2f * Mathf.PI * freq * 2.1f * _time + t * 2f)
                );
            }
        }

        private void GenerateListening()
        {
            // Slightly active sine that reacts to mic input
            float freq  = 1.5f;
            float scale = 0.3f + _smoothedAmplitude * 1.2f;

            for (int i = 0; i < _samples.Length; i++)
            {
                float t = (float)i / (_samples.Length - 1);
                _samples[i] = scale * Mathf.Sin(2f * Mathf.PI * freq * _time + t * 6f);
            }
        }

        private void GenerateThinking()
        {
            // Slow, irregular wobble — deliberation feeling
            float scale = 0.25f;

            for (int i = 0; i < _samples.Length; i++)
            {
                float t = (float)i / (_samples.Length - 1);
                float noise = Mathf.PerlinNoise(t * 3f, _time * _thinkingFreq);
                // Remap PerlinNoise [0,1] → [-1,1]
                _samples[i] = scale * (noise * 2f - 1f);
            }

            // Occasionally shift freq for organic feel
            _thinkingPhase += Time.deltaTime;
            if (_thinkingPhase > 2f)
            {
                _thinkingPhase = 0f;
                _thinkingFreq = UnityEngine.Random.Range(0.4f, 1.4f);
            }
        }

        private void GenerateSpeaking()
        {
            // Sharp reactive waveform driven by live amplitude
            float scale = _smoothedAmplitude * 0.9f + 0.05f;
            float freq  = 8f + _smoothedAmplitude * 12f;

            for (int i = 0; i < _samples.Length; i++)
            {
                float t = (float)i / (_samples.Length - 1);
                // Speech-like: fundamental + harmonics with amplitude envelope
                float envelope = Mathf.Sin(Mathf.PI * t); // ramps up and down
                _samples[i] = scale * envelope * (
                    0.6f * Mathf.Sin(2f * Mathf.PI * freq * t + _time * 3f) +
                    0.3f * Mathf.Sin(2f * Mathf.PI * freq * 2f * t + _time * 5f) +
                    0.1f * Mathf.Sin(2f * Mathf.PI * freq * 3f * t + _time * 7f)
                );
            }
        }

        // ── RENDERING ─────────────────────────────────────────────────────────

        private void RenderSamples()
        {
            if (lineRenderer == null || container == null) return;

            float width  = container.rect.width;
            float height = UIConfig.WaveformHeight;

            for (int i = 0; i < _samples.Length; i++)
            {
                float x = Mathf.Lerp(-width / 2f, width / 2f, (float)i / (_samples.Length - 1));
                float y = _samples[i] * height;
                lineRenderer.SetPosition(i, new Vector3(x, y, 0f));
            }
        }

        // ── THEME ─────────────────────────────────────────────────────────────

        private void ApplyTheme(PersonalityTheme theme)
        {
            if (theme == null || lineRenderer == null) return;
            lineRenderer.startColor = theme.waveformColor;
            lineRenderer.endColor   = theme.waveformColor;
        }

        private void OnThemeChanged(PersonalityTheme theme)
        {
            ApplyTheme(theme);
        }
    }
}
