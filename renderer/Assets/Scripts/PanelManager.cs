// LAVENDER — Panel Manager
// renderer/Assets/Scripts/PanelManager.cs
//
// Manages all floating panels on the holographic display.
// Panels are named by ID — showing the same ID again updates in-place.
//
// Panel types supported:
//   text          — simple text block
//   response_text — Lavender's spoken response, styled prominently
//   key_value     — two-column data display (label: value)
//   data_table    — multi-row table
//   calendar      — meeting/schedule display
//   system_stats  — CPU/RAM/GPU readout with bars
//
// Attach to: an empty GameObject called "PanelManager" under the Canvas.

using System;
using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using UnityEngine.UI;
using TMPro;

namespace Lavender
{
    public class PanelManager : MonoBehaviour
    {
        // ── SINGLETON ────────────────────────────────────────────────────────
        public static PanelManager Instance { get; private set; }

        // ── INSPECTOR ────────────────────────────────────────────────────────

        [Header("Prefabs")]
        [SerializeField] private GameObject panelPrefab;           // Base panel prefab
        [SerializeField] private GameObject textRowPrefab;         // TMP text row
        [SerializeField] private GameObject keyValueRowPrefab;     // label + value row
        [SerializeField] private GameObject statBarRowPrefab;      // stat + bar row

        [Header("Layout Parents — one RectTransform per layer")]
        [SerializeField] private RectTransform layerBackground;    // Layer 0
        [SerializeField] private RectTransform layerPersistent;    // Layer 1
        [SerializeField] private RectTransform layerActive;        // Layer 2
        [SerializeField] private RectTransform layerForeground;    // Layer 3

        [Header("Settings")]
        [SerializeField] private float panelSpacing = 20f;

        // ── INTERNALS ─────────────────────────────────────────────────────────
        private readonly Dictionary<string, PanelInstance> _panels
            = new Dictionary<string, PanelInstance>();

        // ── PANEL DATA MODEL ──────────────────────────────────────────────────
        private class PanelInstance
        {
            public string id;
            public GameObject gameObject;
            public CanvasGroup canvasGroup;
            public TMP_Text titleText;
            public Transform contentRoot;
            public Image backgroundImage;
            public Image topEdgeLine;
            public int layer;
            public bool persist;
            public Coroutine dismissCoroutine;
        }

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
        }

        private void OnDisable()
        {
            ThemeManager.OnThemeChanged -= OnThemeChanged;
        }

        // ── PUBLIC API ────────────────────────────────────────────────────────

        /// <summary>
        /// Show or update a panel. If panel_id already exists, updates content in-place.
        /// </summary>
        public void ShowPanel(PanelDirective directive)
        {
            if (_panels.TryGetValue(directive.panelId, out var existing))
            {
                UpdatePanelContent(existing, directive);
                return;
            }

            CreatePanel(directive);
        }

        /// <summary>
        /// Update only the content of an existing panel.
        /// </summary>
        public void UpdatePanel(string panelId, object content, string title = "")
        {
            if (!_panels.TryGetValue(panelId, out var panel)) return;

            if (!string.IsNullOrEmpty(title) && panel.titleText != null)
                panel.titleText.text = title;

            // Content update delegated per type — for now refresh fully
            // In production, diff and animate only changed rows
        }

        /// <summary>
        /// Dismiss a panel with its exit animation.
        /// </summary>
        public void ClosePanel(string panelId)
        {
            if (!_panels.TryGetValue(panelId, out var panel)) return;
            StartCoroutine(DismissPanel(panel));
        }

        /// <summary>
        /// Remove all non-persistent panels.
        /// </summary>
        public void ClearTransient()
        {
            var toClose = new List<string>();
            foreach (var kvp in _panels)
                if (!kvp.Value.persist)
                    toClose.Add(kvp.Key);

            foreach (var id in toClose)
                ClosePanel(id);
        }

        /// <summary>
        /// Remove everything.
        /// </summary>
        public void ClearAll()
        {
            foreach (var id in new List<string>(_panels.Keys))
                ClosePanel(id);
        }

        // ── PANEL CREATION ────────────────────────────────────────────────────

        private void CreatePanel(PanelDirective d)
        {
            if (panelPrefab == null)
            {
                Debug.LogError("[PanelManager] panelPrefab is not assigned.");
                return;
            }

            RectTransform parent = GetLayerParent(d.layer);

            var go   = Instantiate(panelPrefab, parent);
            var pi   = new PanelInstance
            {
                id          = d.panelId,
                gameObject  = go,
                canvasGroup = go.GetComponent<CanvasGroup>() ?? go.AddComponent<CanvasGroup>(),
                layer       = d.layer,
                persist     = d.persist,
            };

            // Get references to child components
            pi.titleText    = go.transform.Find("Title")?.GetComponent<TMP_Text>();
            pi.contentRoot  = go.transform.Find("Content");
            pi.backgroundImage = go.GetComponent<Image>();
            pi.topEdgeLine  = go.transform.Find("TopEdge")?.GetComponent<Image>();

            // Apply current theme
            ApplyTheme(pi, ThemeManager.Current);

            // Set title
            if (pi.titleText != null && !string.IsNullOrEmpty(d.title))
            {
                pi.titleText.text = d.title;
                pi.titleText.gameObject.SetActive(true);
            }
            else if (pi.titleText != null)
            {
                pi.titleText.gameObject.SetActive(false);
            }

            // Populate content
            PopulateContent(pi, d);

            _panels[d.panelId] = pi;

            // Animate in
            StartCoroutine(AnimateIn(pi, d.animateIn));

            // Auto-dismiss
            if (d.autoDismiss > 0f)
            {
                pi.dismissCoroutine = StartCoroutine(AutoDismiss(pi, d.autoDismiss));
            }
        }

        // ── CONTENT POPULATION ────────────────────────────────────────────────

        private void PopulateContent(PanelInstance pi, PanelDirective d)
        {
            if (pi.contentRoot == null) return;

            // Clear existing content
            foreach (Transform child in pi.contentRoot)
                Destroy(child.gameObject);

            switch (d.contentType)
            {
                case "text":
                case "response_text":
                    PopulateText(pi, d.text, d.contentType == "response_text");
                    break;

                case "key_value":
                    PopulateKeyValue(pi, d.keyValueData);
                    break;

                case "system_stats":
                    PopulateSystemStats(pi, d.keyValueData);
                    break;

                case "calendar":
                    PopulateCalendar(pi, d.text, d.keyValueData);
                    break;

                case "data_table":
                    PopulateDataTable(pi, d.text);
                    break;

                default:
                    PopulateText(pi, d.text, false);
                    break;
            }
        }

        private void PopulateText(PanelInstance pi, string text, bool isResponse)
        {
            if (textRowPrefab == null || string.IsNullOrEmpty(text)) return;

            var row = Instantiate(textRowPrefab, pi.contentRoot);
            var tmp = row.GetComponent<TMP_Text>();
            if (tmp == null) return;

            tmp.text = text;
            tmp.fontSize = isResponse ? UIConfig.FontSizeBody * 1.1f : UIConfig.FontSizeBody;
            tmp.color = UIConfig.WithAlpha(
                ThemeManager.Current.primaryColor,
                isResponse ? 1.0f : ThemeManager.Current.unfocusedTextOpacity + 0.2f
            );

            if (isResponse)
                tmp.fontStyle = FontStyles.Normal;
        }

        private void PopulateKeyValue(PanelInstance pi, Dictionary<string, string> data)
        {
            if (data == null || keyValueRowPrefab == null) return;

            foreach (var kvp in data)
            {
                var row = Instantiate(keyValueRowPrefab, pi.contentRoot);
                var texts = row.GetComponentsInChildren<TMP_Text>();
                if (texts.Length >= 2)
                {
                    texts[0].text  = kvp.Key;
                    texts[0].color = UIConfig.WithAlpha(
                        ThemeManager.Current.primaryColor,
                        ThemeManager.Current.unfocusedTextOpacity
                    );
                    texts[1].text  = kvp.Value;
                    texts[1].color = ThemeManager.Current.primaryColor;
                }
            }
        }

        private void PopulateSystemStats(PanelInstance pi, Dictionary<string, string> data)
        {
            if (data == null || statBarRowPrefab == null) return;

            foreach (var kvp in data)
            {
                var row = Instantiate(statBarRowPrefab, pi.contentRoot);
                var texts   = row.GetComponentsInChildren<TMP_Text>();
                var sliders = row.GetComponentsInChildren<Slider>();

                if (texts.Length >= 2)
                {
                    texts[0].text = kvp.Key;
                    float.TryParse(kvp.Value.Replace("%", ""), out float pct);
                    texts[1].text = kvp.Value;

                    if (sliders.Length > 0)
                    {
                        sliders[0].value = pct / 100f;
                        var fill = sliders[0].fillRect?.GetComponent<Image>();
                        if (fill != null)
                        {
                            // Color the bar based on threshold
                            fill.color = pct > 85f
                                ? new Color(1f, 0.27f, 0.27f)   // red
                                : pct > 65f
                                    ? new Color(1f, 0.70f, 0.28f)  // orange
                                    : ThemeManager.Current.accentColor;
                        }
                    }
                }
            }
        }

        private void PopulateCalendar(PanelInstance pi, string nextEvent,
                                      Dictionary<string, string> details)
        {
            // Next event display
            if (!string.IsNullOrEmpty(nextEvent))
                PopulateText(pi, nextEvent, false);

            if (details != null)
                PopulateKeyValue(pi, details);
        }

        private void PopulateDataTable(PanelInstance pi, string json)
        {
            // Deserialize JSON array of row strings and display
            try
            {
                var rows = Newtonsoft.Json.JsonConvert.DeserializeObject<string[]>(json);
                foreach (var row in rows)
                    PopulateText(pi, row, false);
            }
            catch
            {
                PopulateText(pi, json, false);
            }
        }

        // ── THEME APPLICATION ─────────────────────────────────────────────────

        private void ApplyTheme(PanelInstance pi, PersonalityTheme theme)
        {
            if (pi.backgroundImage != null)
                pi.backgroundImage.color = theme.panelBackground;

            if (pi.topEdgeLine != null)
                pi.topEdgeLine.color = UIConfig.WithAlpha(theme.primaryColor, UIConfig.PanelEdgeOpacity);

            if (pi.titleText != null)
                pi.titleText.color = UIConfig.WithAlpha(theme.primaryColor, 0.7f);
        }

        private void OnThemeChanged(PersonalityTheme theme)
        {
            foreach (var panel in _panels.Values)
                ApplyTheme(panel, theme);
        }

        // ── ANIMATIONS ────────────────────────────────────────────────────────

        private IEnumerator AnimateIn(PanelInstance pi, string style)
        {
            var cg = pi.canvasGroup;
            var rt = pi.gameObject.GetComponent<RectTransform>();
            if (cg == null) yield break;

            float duration = ThemeManager.Current?.animationDuration ?? 0.25f;
            style = style ?? ThemeManager.Current?.panelEnterStyle ?? "fade";

            switch (style)
            {
                case "instant":
                    cg.alpha = 1f;
                    break;

                case "fade":
                    yield return FadeIn(cg, duration);
                    break;

                case "slide":
                    if (rt != null)
                    {
                        Vector2 start = rt.anchoredPosition + new Vector2(60f, 0f);
                        Vector2 end   = rt.anchoredPosition;
                        yield return SlideIn(rt, cg, start, end, duration);
                    }
                    else yield return FadeIn(cg, duration);
                    break;

                case "materialize":
                    yield return Materialize(cg, duration);
                    break;

                default:
                    yield return FadeIn(cg, duration);
                    break;
            }
        }

        private IEnumerator FadeIn(CanvasGroup cg, float duration)
        {
            float elapsed = 0f;
            cg.alpha = 0f;

            while (elapsed < duration)
            {
                cg.alpha = Mathf.Lerp(0f, 1f, elapsed / duration);
                elapsed += Time.deltaTime;
                yield return null;
            }
            cg.alpha = 1f;
        }

        private IEnumerator SlideIn(RectTransform rt, CanvasGroup cg,
                                    Vector2 from, Vector2 to, float duration)
        {
            float elapsed = 0f;
            cg.alpha = 0f;
            rt.anchoredPosition = from;

            while (elapsed < duration)
            {
                float t = elapsed / duration;
                float eased = 1f - Mathf.Pow(1f - t, 3f); // ease out cubic
                rt.anchoredPosition = Vector2.Lerp(from, to, eased);
                cg.alpha = Mathf.Lerp(0f, 1f, t);
                elapsed += Time.deltaTime;
                yield return null;
            }

            rt.anchoredPosition = to;
            cg.alpha = 1f;
        }

        private IEnumerator Materialize(CanvasGroup cg, float duration)
        {
            // Static flicker then snap to full opacity — Lilac's signature
            cg.alpha = 0f;
            yield return new WaitForSeconds(0.05f);
            cg.alpha = 0.3f;
            yield return new WaitForSeconds(0.04f);
            cg.alpha = 0.1f;
            yield return new WaitForSeconds(0.06f);
            cg.alpha = 0.7f;
            yield return new WaitForSeconds(0.03f);
            cg.alpha = 1f;
        }

        private IEnumerator DismissPanel(PanelInstance pi)
        {
            var cg = pi.canvasGroup;
            if (cg != null)
            {
                float elapsed  = 0f;
                float duration = 0.15f;
                while (elapsed < duration)
                {
                    cg.alpha = Mathf.Lerp(1f, 0f, elapsed / duration);
                    elapsed += Time.deltaTime;
                    yield return null;
                }
            }

            _panels.Remove(pi.id);
            Destroy(pi.gameObject);
        }

        private IEnumerator AutoDismiss(PanelInstance pi, float delay)
        {
            yield return new WaitForSeconds(delay);
            if (_panels.ContainsKey(pi.id))
                ClosePanel(pi.id);
        }

        // ── HELPERS ───────────────────────────────────────────────────────────

        private RectTransform GetLayerParent(int layer)
        {
            return layer switch
            {
                0 => layerBackground ?? (RectTransform)transform,
                1 => layerPersistent ?? (RectTransform)transform,
                2 => layerActive     ?? (RectTransform)transform,
                3 => layerForeground ?? (RectTransform)transform,
                _ => (RectTransform)transform,
            };
        }

        private void UpdatePanelContent(PanelInstance pi, PanelDirective d)
        {
            if (!string.IsNullOrEmpty(d.title) && pi.titleText != null)
                pi.titleText.text = d.title;

            PopulateContent(pi, d);

            // Reset auto-dismiss timer if applicable
            if (pi.dismissCoroutine != null)
                StopCoroutine(pi.dismissCoroutine);

            if (d.autoDismiss > 0f)
                pi.dismissCoroutine = StartCoroutine(AutoDismiss(pi, d.autoDismiss));
        }
    }

    // ── DATA STRUCTURES ───────────────────────────────────────────────────────

    [System.Serializable]
    public class PanelDirective
    {
        public string panelId;
        public int    layer;
        public string contentType;
        public string text;
        public Dictionary<string, string> keyValueData;
        public string title;
        public string animateIn;
        public bool   persist;
        public float  autoDismiss;
        public string personality;
    }
}
