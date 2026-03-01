// LAVENDER — Director Client
// renderer/Assets/Scripts/LavenderDirectorClient.cs
//
// THE main Unity entry point for the hologram renderer.
// Connects to the Python WebSocket server (HologramDirector).
// Receives JSON directives and routes them to the right subsystem.
//
// Setup:
//   1. Create empty GameObject "LavenderSystem" in your scene
//   2. Attach this script
//   3. Assign ThemeManager, PanelManager, WaveformRenderer, AmbientDisplay
//      references in the Inspector
//   4. Import NativeWebSocket: https://github.com/endel/NativeWebSocket
//      (Package Manager → Add from git URL)
//
// The renderer should be its own Unity scene running in fullscreen
// on the projector output (Display 2). Main monitor runs your OS normally.

using System;
using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using NativeWebSocket;
using Newtonsoft.Json;
using Newtonsoft.Json.Linq;

namespace Lavender
{
    public class LavenderDirectorClient : MonoBehaviour
    {
        // ── INSPECTOR ────────────────────────────────────────────────────────

        [Header("Connection")]
        [SerializeField] private string serverUrl  = "ws://localhost:8765";
        [SerializeField] private float  reconnectDelay = 3f;

        [Header("Subsystem References")]
        [SerializeField] private ThemeManager    themeManager;
        [SerializeField] private PanelManager    panelManager;
        [SerializeField] private WaveformRenderer waveformRenderer;
        [SerializeField] private AmbientDisplay   ambientDisplay;

        [Header("Debug")]
        [SerializeField] private bool logDirectives = false;

        // ── INTERNALS ─────────────────────────────────────────────────────────

        private WebSocket  _ws;
        private bool       _running = true;
        private bool       _connected = false;

        // Thread-safe queue for directives received from WebSocket
        // WebSocket callbacks run on a background thread;
        // Unity APIs must be called from the main thread.
        private readonly Queue<string> _messageQueue = new Queue<string>();
        private readonly object        _queueLock    = new object();

        // ── UNITY ────────────────────────────────────────────────────────────

        private void Awake()
        {
            // Auto-find subsystems if not assigned in inspector
            if (themeManager    == null) themeManager    = FindObjectOfType<ThemeManager>();
            if (panelManager    == null) panelManager    = FindObjectOfType<PanelManager>();
            if (waveformRenderer == null) waveformRenderer = FindObjectOfType<WaveformRenderer>();
            if (ambientDisplay  == null) ambientDisplay  = FindObjectOfType<AmbientDisplay>();
        }

        private void Start()
        {
            StartCoroutine(ConnectLoop());
        }

        private void Update()
        {
            // Dispatch queued WebSocket messages on the main thread
            lock (_queueLock)
            {
                while (_messageQueue.Count > 0)
                {
                    var msg = _messageQueue.Dequeue();
                    try { HandleMessage(msg); }
                    catch (Exception e)
                    {
                        Debug.LogError($"[LavenderClient] Error handling message: {e}");
                    }
                }
            }

            // NativeWebSocket requires this dispatch call
#if !UNITY_WEBGL || UNITY_EDITOR
            _ws?.DispatchMessageQueue();
#endif
        }

        private void OnDestroy()
        {
            _running = false;
            _ws?.Close();
        }

        // ── CONNECTION ────────────────────────────────────────────────────────

        private IEnumerator ConnectLoop()
        {
            while (_running)
            {
                yield return StartCoroutine(Connect());

                if (_running)
                {
                    Debug.Log($"[LavenderClient] Reconnecting in {reconnectDelay}s...");
                    yield return new WaitForSeconds(reconnectDelay);
                }
            }
        }

        private IEnumerator Connect()
        {
            Debug.Log($"[LavenderClient] Connecting to {serverUrl}...");

            _ws = new WebSocket(serverUrl);

            _ws.OnOpen += () =>
            {
                _connected = true;
                Debug.Log("[LavenderClient] Connected to Lavender director.");
            };

            _ws.OnMessage += (bytes) =>
            {
                var message = System.Text.Encoding.UTF8.GetString(bytes);
                lock (_queueLock)
                    _messageQueue.Enqueue(message);
            };

            _ws.OnError += (error) =>
            {
                Debug.LogWarning($"[LavenderClient] WebSocket error: {error}");
            };

            _ws.OnClose += (code) =>
            {
                _connected = false;
                Debug.Log($"[LavenderClient] Disconnected (code: {code})");
            };

            yield return _ws.Connect();

            // Keep alive until disconnected
            while (_ws.State == WebSocketState.Open && _running)
                yield return null;
        }

        // ── MESSAGE ROUTING ───────────────────────────────────────────────────

        private void HandleMessage(string json)
        {
            if (logDirectives)
                Debug.Log($"[LavenderClient] ← {json}");

            JObject msg;
            try { msg = JObject.Parse(json); }
            catch { Debug.LogWarning($"[LavenderClient] Malformed JSON: {json}"); return; }

            string type    = msg["type"]?.ToString() ?? "";
            JToken payload = msg["payload"];

            switch (type)
            {
                // ── THEME ─────────────────────────────────────────────────────
                case "set_theme":
                    HandleSetTheme(payload);
                    break;

                case "personality_transition":
                    HandlePersonalityTransition(payload);
                    break;

                // ── STATE ─────────────────────────────────────────────────────
                case "set_state":
                    HandleSetState(payload);
                    break;

                case "set_brightness":
                    HandleSetBrightness(payload);
                    break;

                // ── PANELS ────────────────────────────────────────────────────
                case "show_panel":
                    HandleShowPanel(payload);
                    break;

                case "update_panel":
                    HandleUpdatePanel(payload);
                    break;

                case "close_panel":
                    HandleClosePanel(payload);
                    break;

                case "clear_all":
                    panelManager?.ClearAll();
                    break;

                // ── WAVEFORM ──────────────────────────────────────────────────
                case "set_waveform":
                    HandleSetWaveform(payload);
                    break;

                case "push_audio":
                    HandlePushAudio(payload);
                    break;

                // ── AMBIENT ───────────────────────────────────────────────────
                case "update_clock":
                    HandleUpdateClock(payload);
                    break;

                case "update_ambient":
                    HandleUpdateAmbient(payload);
                    break;

                case "show_alert":
                    HandleShowAlert(payload);
                    break;

                default:
                    Debug.LogWarning($"[LavenderClient] Unknown directive type: '{type}'");
                    break;
            }
        }

        // ── HANDLERS ──────────────────────────────────────────────────────────

        private void HandleSetTheme(JToken p)
        {
            string personality = p?["theme"]?.ToString() ?? "nova";
            themeManager?.ApplyInstant(personality);
        }

        private void HandlePersonalityTransition(JToken p)
        {
            string personality  = p?["theme"]?.ToString() ?? "nova";
            float  durationMs   = p?["duration_ms"]?.Value<float>() ?? 1800f;
            themeManager?.TransitionTo(personality, durationMs);
        }

        private void HandleSetState(JToken p)
        {
            string state = p?["state"]?.ToString() ?? "ambient";
            ambientDisplay?.SetState(state);

            // Focus mode side effect
            if (state == "focus")
                ambientDisplay?.SetFocusMode(true);
            else if (state == "ambient" || state == "active")
                ambientDisplay?.SetFocusMode(false);

            waveformRenderer?.SetState(state == "thinking" ? "thinking" : "idle");
        }

        private void HandleSetBrightness(JToken p)
        {
            float value = p?["value"]?.Value<float>() ?? 1f;
            themeManager?.SetBrightness(value);
        }

        private void HandleShowPanel(JToken p)
        {
            if (p == null || panelManager == null) return;

            var directive = new PanelDirective
            {
                panelId     = p["panel_id"]?.ToString()     ?? "panel_" + Time.time,
                layer       = p["layer"]?.Value<int>()       ?? 2,
                contentType = p["content_type"]?.ToString()  ?? "text",
                text        = p["text"]?.ToString()          ?? p["content"]?.ToString() ?? "",
                title       = p["title"]?.ToString()         ?? "",
                animateIn   = p["animate_in"]?.ToString()    ?? "fade",
                persist     = p["persist"]?.Value<bool>()    ?? true,
                autoDismiss = p["auto_dismiss"]?.Value<float>() ?? 0f,
                personality = p["personality"]?.ToString()   ?? "nova",
            };

            // Parse key_value data if present
            var contentToken = p["content"];
            if (contentToken?.Type == JTokenType.Object)
            {
                directive.keyValueData = contentToken.ToObject<Dictionary<string, string>>();
                directive.text = "";
            }

            panelManager.ShowPanel(directive);

            // If this is a response panel, also set waveform to speaking
            if (directive.contentType == "response_text")
                waveformRenderer?.SetState("speaking");
        }

        private void HandleUpdatePanel(JToken p)
        {
            if (p == null || panelManager == null) return;

            string panelId = p["panel_id"]?.ToString() ?? "";
            string content = p["content"]?.ToString() ?? "";
            string title   = p["title"]?.ToString() ?? "";

            panelManager.UpdatePanel(panelId, content, title);
        }

        private void HandleClosePanel(JToken p)
        {
            string panelId = p?["panel_id"]?.ToString() ?? "";
            if (!string.IsNullOrEmpty(panelId))
                panelManager?.ClosePanel(panelId);
        }

        private void HandleSetWaveform(JToken p)
        {
            string state     = p?["state"]?.ToString() ?? "idle";
            float  amplitude = p?["amplitude"]?.Value<float>() ?? 0f;
            waveformRenderer?.SetState(state, amplitude);
        }

        private void HandlePushAudio(JToken p)
        {
            float amplitude = p?["amplitude"]?.Value<float>() ?? 0f;
            waveformRenderer?.PushAmplitude(amplitude);
        }

        private void HandleUpdateClock(JToken p)
        {
            string timeStr = p?["time"]?.ToString() ?? "";
            string dateStr = p?["date"]?.ToString() ?? "";
            ambientDisplay?.UpdateClock(timeStr, dateStr);
        }

        private void HandleUpdateAmbient(JToken p)
        {
            if (p == null || ambientDisplay == null) return;

            float cpu = p["cpu_pct"]?.Value<float>() ?? 0f;
            float ram = p["ram_pct"]?.Value<float>() ?? 0f;
            float gpu = p["gpu_pct"]?.Value<float>() ?? 0f;

            ambientDisplay.UpdateStats(cpu, ram, gpu);
        }

        private void HandleShowAlert(JToken p)
        {
            if (p == null || panelManager == null) return;

            string title    = p["title"]?.ToString()    ?? "Alert";
            string message  = p["message"]?.ToString()  ?? "";
            string severity = p["severity"]?.ToString() ?? "info";

            // Alert panels use Layer 3 (foreground), never auto-dismiss
            panelManager.ShowPanel(new PanelDirective
            {
                panelId     = "alert_" + Time.time,
                layer       = 3,
                contentType = "text",
                title       = title,
                text        = message,
                animateIn   = "fade",
                persist     = false,
                autoDismiss = severity == "info" ? 8f : 0f,   // critical stays until dismissed
            });
        }

        // ── STATUS ────────────────────────────────────────────────────────────

        public bool IsConnected => _connected;
    }
}
