// LAVENDER — Theme Manager
// renderer/Assets/Scripts/ThemeManager.cs
//
// Receives personality theme changes and applies them across the entire UI.
// Handles the collapse → hold → expand transition animation.
// All other scripts reference ThemeManager.Current for live color values.
//
// Attach to: a single empty GameObject called "ThemeManager" in the scene.

using System;
using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using UnityEngine.UI;
using TMPro;

namespace Lavender
{
    public class ThemeManager : MonoBehaviour
    {
        // ── SINGLETON ────────────────────────────────────────────────────────
        public static ThemeManager Instance { get; private set; }

        // Current live theme — all other scripts read from here
        public static PersonalityTheme Current { get; private set; }

        // Event fired after a theme transition completes
        public static event Action<PersonalityTheme> OnThemeChanged;

        // ── INSPECTOR ────────────────────────────────────────────────────────

        [Header("Scene References")]
        [SerializeField] private CanvasGroup masterCanvasGroup;
        [SerializeField] private Image backgroundPanel;

        [Header("State")]
        [SerializeField] private string currentPersonality = "nova";
        [SerializeField] private bool isTransitioning = false;

        // ── UNITY ────────────────────────────────────────────────────────────

        private void Awake()
        {
            if (Instance != null && Instance != this)
            {
                Destroy(gameObject);
                return;
            }
            Instance = this;
            DontDestroyOnLoad(gameObject);

            // Apply default theme immediately on start
            Current = UIConfig.GetTheme(currentPersonality);
            ApplyInstant(Current);
        }

        // ── PUBLIC API — called by LavenderDirectorClient ─────────────────

        /// <summary>
        /// Immediately apply a theme without transition animation.
        /// Used on initial load and renderer reconnect.
        /// </summary>
        public void ApplyInstant(string personality)
        {
            var theme = UIConfig.GetTheme(personality);
            currentPersonality = personality;
            Current = theme;
            ApplyInstant(theme);
            OnThemeChanged?.Invoke(theme);
        }

        private void ApplyInstant(PersonalityTheme theme)
        {
            if (backgroundPanel != null)
                backgroundPanel.color = theme.backgroundColor;

            if (masterCanvasGroup != null)
                masterCanvasGroup.alpha = 1.0f;
        }

        /// <summary>
        /// Full animated personality transition.
        /// Collapses inward → holds → expands with new theme.
        /// Duration: ~1.8 seconds total.
        /// </summary>
        public void TransitionTo(string personality, float durationMs = 1800f)
        {
            if (isTransitioning) return;
            if (personality == currentPersonality) return;

            var newTheme = UIConfig.GetTheme(personality);
            StartCoroutine(RunTransition(newTheme, durationMs / 1000f));
        }

        // ── TRANSITION COROUTINE ──────────────────────────────────────────

        private IEnumerator RunTransition(PersonalityTheme newTheme, float totalDuration)
        {
            isTransitioning = true;

            float collapseTime = UIConfig.TransitionCollapseDuration;
            float holdTime     = UIConfig.TransitionHoldDuration;
            float expandTime   = UIConfig.TransitionExpandDuration;

            // ── PHASE 1: COLLAPSE — fade out + scale down
            yield return StartCoroutine(CollapsePhase(collapseTime));

            // ── PHASE 2: HOLD — brief darkness, swap theme
            Current = newTheme;
            currentPersonality = newTheme.name;

            if (backgroundPanel != null)
                backgroundPanel.color = newTheme.backgroundColor;

            yield return new WaitForSeconds(holdTime);

            // Notify all subscribers DURING the hold so they can update colors
            // before the expand reveals them
            OnThemeChanged?.Invoke(newTheme);

            // ── PHASE 3: EXPAND — fade in with new theme
            yield return StartCoroutine(ExpandPhase(expandTime));

            isTransitioning = false;
        }

        private IEnumerator CollapsePhase(float duration)
        {
            if (masterCanvasGroup == null) yield break;

            float elapsed = 0f;
            float startAlpha = masterCanvasGroup.alpha;

            // Fade out + subtle scale down using transform
            Vector3 startScale = transform.localScale;
            Vector3 endScale   = Vector3.one * 0.92f;

            while (elapsed < duration)
            {
                float t = elapsed / duration;
                float eased = EaseInCubic(t);

                masterCanvasGroup.alpha = Mathf.Lerp(startAlpha, 0f, eased);

                if (transform != null)
                    transform.localScale = Vector3.Lerp(startScale, endScale, eased);

                elapsed += Time.deltaTime;
                yield return null;
            }

            masterCanvasGroup.alpha = 0f;
        }

        private IEnumerator ExpandPhase(float duration)
        {
            if (masterCanvasGroup == null) yield break;

            float elapsed = 0f;
            Vector3 startScale = transform.localScale;
            Vector3 endScale   = Vector3.one;

            while (elapsed < duration)
            {
                float t = elapsed / duration;
                float eased = EaseOutCubic(t);

                masterCanvasGroup.alpha = Mathf.Lerp(0f, 1f, eased);

                if (transform != null)
                    transform.localScale = Vector3.Lerp(startScale, endScale, eased);

                elapsed += Time.deltaTime;
                yield return null;
            }

            masterCanvasGroup.alpha = 1f;
            transform.localScale    = Vector3.one;
        }

        // ── BRIGHTNESS ────────────────────────────────────────────────────

        public void SetBrightness(float value)
        {
            if (masterCanvasGroup != null)
                masterCanvasGroup.alpha = Mathf.Clamp01(value);
        }

        // ── EASING ────────────────────────────────────────────────────────

        private static float EaseInCubic(float t)  => t * t * t;
        private static float EaseOutCubic(float t)
        {
            t = 1f - t;
            return 1f - (t * t * t);
        }

        // ── STATE QUERY ───────────────────────────────────────────────────

        public string CurrentPersonality => currentPersonality;
        public bool IsTransitioning => isTransitioning;

        /// <summary>
        /// Apply current theme colors to a text element.
        /// </summary>
        public static void StyleText(TMP_Text text, bool isFocused = false)
        {
            if (Current == null || text == null) return;

            float alpha = isFocused ? 1.0f : Current.unfocusedTextOpacity;
            text.color = UIConfig.WithAlpha(Current.primaryColor, alpha);
        }

        /// <summary>
        /// Apply current theme colors to an accent element (buttons, lines).
        /// </summary>
        public static void StyleAccent(Graphic graphic)
        {
            if (Current == null || graphic == null) return;
            graphic.color = Current.accentColor;
        }
    }
}
