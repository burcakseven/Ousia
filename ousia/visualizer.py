"""
Ousia Visualizer — Geçici Görselleştirme Modülü

NetworkX + Matplotlib kullanır. Sunucu veya tarayıcı gerektirmez.

Kullanım:
    from visualizer import OusiaVisualizer
    viz = OusiaVisualizer(graph)
    viz.show()          # interaktif pencere
    viz.save("out.png") # dosyaya kaydet

    # Ya da tek satırda:
    from visualizer import quick_plot
    quick_plot(graph)
"""

from typing import Optional
from .graph_engine import PatientGraph

try:
    import networkx as nx
    import matplotlib
    import matplotlib.pyplot as plt
    import matplotlib.cm as cm
    import matplotlib.colors as mcolors
    import numpy as np
    _DEPS_OK = True
except ImportError as _e:
    _DEPS_OK = False
    _MISSING = str(_e)


class OusiaVisualizer:
    """
    PatientGraph için görselleştirici.

    Görsel kodlama:
      - Düğüm boyutu   → activation_count (ne kadar sık aktive olduysa o kadar büyük)
      - Düğüm rengi    → açık mavi (normal) | koyu gri (silence düğümü)
      - Kenar kalınlığı → weight (bağ gücü)
      - Kenar rengi    → Consonance (Yeşil) vs Dissonance (Kırmızı) dengesi
      - Kenar stili    → silence kenarları: kesik çizgi (dashed)
    """

    NODE_SIZE_BASE    = 600
    NODE_SIZE_SCALE   = 300
    EDGE_WIDTH_SCALE  = 6.0
    FONT_SIZE         = 9
    FIG_SIZE          = (13, 9)
    DPI               = 130

    def __init__(self, graph: PatientGraph):
        if not _DEPS_OK:
            raise ImportError(
                f"Görselleştirme için eksik paket: {_MISSING}\n"
                "Yüklemek için:\n  pip install networkx matplotlib"
            )
        self.graph = graph
        self._G: Optional["nx.Graph"] = None

    # ── İç yardımcılar ────────────────────────────────────────────────────────

    def _build_nx(self) -> "nx.Graph":
        """PatientGraph → NetworkX Graph dönüşümü."""
        G = nx.Graph()
        g = self.graph

        # Düğümleri ekle
        for node in g.nodes:
            G.add_node(node)

        # Kenarları ekle (ağırlık, dissonance, silence flag)
        for key, edge in g.edges.items():
            nodes = list(key)
            if len(nodes) != 2:
                continue
            a, b = nodes[0], nodes[1]
            is_silence = "silence" in key
            G.add_edge(
                a, b,
                weight=edge.weight,
                activation_count=edge.activation_count,
                dissonance=edge.dissonance,
                consonance=edge.consonance,
                intensity=edge.emotional_intensity,
                is_silence=is_silence,
            )

        self._G = G
        return G

    def _node_activation_count(self, G: "nx.Graph") -> dict:
        """Her düğüm için en yüksek komşu kenar activation_count'unu döner."""
        counts = {n: 0 for n in G.nodes}
        for u, v, data in G.edges(data=True):
            ac = data.get("activation_count", 0)
            counts[u] = max(counts[u], ac)
            counts[v] = max(counts[v], ac)
        return counts

    @staticmethod
    def _edge_color(d: float, c: float) -> str:
        """Dissonance (red) vs Consonance (green) balancing."""
        if d > c and d > 0.3:
            # Shift towards red
            cmap = matplotlib.colormaps["YlOrRd"]
            rgba = cmap(float(np.clip(d, 0.0, 1.0)))
        elif c >= d and c > 0.3:
            # Shift towards green
            cmap = matplotlib.colormaps["YlGn"]
            rgba = cmap(float(np.clip(c, 0.0, 1.0)))
        else:
            # Neutral / low intensity
            return "#666688"
        return mcolors.to_hex(rgba)

    # ── Ana render ────────────────────────────────────────────────────────────

    def _render(self) -> "plt.Figure":
        G = self._build_nx()

        if len(G.nodes) == 0:
            fig, ax = plt.subplots(figsize=self.FIG_SIZE)
            ax.text(0.5, 0.5, "Graf boş — henüz veri yok.",
                    ha="center", va="center", fontsize=14, color="gray")
            ax.axis("off")
            fig.patch.set_facecolor("#1a1a2e")
            return fig

        # Layout: spring (ağırlıklı), silence düğümü varsa çevreye çekiliyor
        try:
            pos = nx.spring_layout(G, weight="weight", seed=42, k=2.5)
        except Exception:
            pos = nx.shell_layout(G)

        # ── Düğüm özellikleri ───────────────────────────────────────────────
        act_counts = self._node_activation_count(G)
        node_list  = list(G.nodes)
        node_sizes = [
            self.NODE_SIZE_BASE + self.NODE_SIZE_SCALE * act_counts.get(n, 0)
            for n in node_list
        ]
        node_colors = [
            "#3a3a5c" if n == "silence" else "#4a90d9"
            for n in node_list
        ]

        # ── Kenar grupları ───────────────────────────────────────────────────
        normal_edges   = []
        silence_edges  = []
        normal_widths  = []
        silence_widths = []
        normal_colors  = []

        for u, v, data in G.edges(data=True):
            w  = data.get("weight", 0.0)
            d  = data.get("dissonance", 0.0)
            c  = data.get("consonance", 0.0)
            is_sil = data.get("is_silence", False)
            width = max(0.5, w * self.EDGE_WIDTH_SCALE)

            if is_sil:
                silence_edges.append((u, v))
                silence_widths.append(width * 0.8)
            else:
                normal_edges.append((u, v))
                normal_widths.append(width)
                normal_colors.append(self._edge_color(d, c))

        # ── Figür ────────────────────────────────────────────────────────────
        fig, ax = plt.subplots(figsize=self.FIG_SIZE, dpi=self.DPI)
        fig.patch.set_facecolor("#1a1a2e")
        ax.set_facecolor("#1a1a2e")

        # Normal kenarlar (dissonance renkli)
        if normal_edges:
            nx.draw_networkx_edges(
                G, pos,
                edgelist=normal_edges,
                width=normal_widths,
                edge_color=normal_colors,
                alpha=0.75,
                ax=ax,
            )

        # Silence kenarları (kesik çizgi, gri)
        if silence_edges:
            nx.draw_networkx_edges(
                G, pos,
                edgelist=silence_edges,
                width=silence_widths,
                edge_color="#888888",
                style="dashed",
                alpha=0.6,
                ax=ax,
            )

        # Düğümler
        nx.draw_networkx_nodes(
            G, pos,
            nodelist=node_list,
            node_size=node_sizes,
            node_color=node_colors,
            alpha=0.92,
            linewidths=1.5,
            edgecolors="#aaaacc",
            ax=ax,
        )

        # Etiketler
        nx.draw_networkx_labels(
            G, pos,
            font_size=self.FONT_SIZE,
            font_color="#f0f0f0",
            font_weight="bold",
            ax=ax,
        )

        # Kenar ağırlık etiketleri (yalnızca w > 0.15 olanlar, fazla kalabalık olmasın)
        edge_labels = {
            (u, v): f"{data['weight']:.2f}"
            for u, v, data in G.edges(data=True)
            if data.get("weight", 0) > 0.15
        }
        nx.draw_networkx_edge_labels(
            G, pos,
            edge_labels=edge_labels,
            font_size=7,
            font_color="#cccccc",
            ax=ax,
        )

        # ── Başlık ve legend ─────────────────────────────────────────────────
        n_nodes = len(G.nodes)
        n_edges = len(G.edges)
        ax.set_title(
            f"Ousia — Psikolojik Kavram Grafı\n"
            f"{n_nodes} kavram · {n_edges} bağ",
            color="#e0e0ff",
            fontsize=12,
            fontweight="bold",
            pad=12,
        )
        ax.axis("off")

        # Skalalar
        # Dissonance Scale (Red)
        sm_d = plt.cm.ScalarMappable(cmap=matplotlib.colormaps["YlOrRd"], norm=mcolors.Normalize(vmin=0, vmax=1))
        sm_d.set_array([])
        cbar_d = fig.colorbar(sm_d, ax=ax, fraction=0.02, pad=0.02, shrink=0.5)
        cbar_d.set_label("Çelişki (Dissonance)", color="#ffaaaa", fontsize=8)
        cbar_d.ax.yaxis.set_tick_params(color="#cccccc", labelcolor="#cccccc")

        # Consonance Scale (Green)
        sm_c = plt.cm.ScalarMappable(cmap=matplotlib.colormaps["YlGn"], norm=mcolors.Normalize(vmin=0, vmax=1))
        sm_c.set_array([])
        cbar_c = fig.colorbar(sm_c, ax=ax, fraction=0.02, pad=0.05, shrink=0.5)
        cbar_c.set_label("Uyum (Consonance)", color="#aaffaa", fontsize=8)
        cbar_c.ax.yaxis.set_tick_params(color="#cccccc", labelcolor="#cccccc")

        # Küçük legend
        from matplotlib.lines import Line2D
        legend_elems = [
            Line2D([0], [0], color="#888888", linestyle="--", linewidth=1.5,
                   label="Sessizlik / Kaçınma kenarı"),
            Line2D([0], [0], marker="o", color="w", markerfacecolor="#4a90d9",
                   markersize=9, label="Kavram düğümü"),
            Line2D([0], [0], marker="o", color="w", markerfacecolor="#3a3a5c",
                   markersize=9, label="Sessizlik düğümü"),
        ]
        ax.legend(
            handles=legend_elems,
            loc="lower left",
            facecolor="#252545",
            edgecolor="#555577",
            labelcolor="#cccccc",
            fontsize=8,
        )

        plt.tight_layout()
        return fig

    # ── Genel API ─────────────────────────────────────────────────────────────

    def show(self):
        """İnteraktif Matplotlib penceresi aç."""
        fig = self._render()
        plt.show()
        plt.close(fig)

    def save(self, path: str = "ousia_graph.png"):
        """Grafı PNG (veya PDF/SVG) olarak kaydet."""
        fig = self._render()
        fig.savefig(path, dpi=self.DPI, bbox_inches="tight", facecolor=fig.get_facecolor())
        plt.close(fig)
        print(f"✓ Graf kaydedildi: {path}")
        return path

    def summary(self):
        """Terminal'e kısa özet yaz."""
        self.graph.summary()


# ── Kısayol fonksiyonu ────────────────────────────────────────────────────────

def quick_plot(graph: PatientGraph, save_path: Optional[str] = None):
    """
    Tek satırda görselleştir.

    save_path verilirse PNG kaydeder, verilmezse pencere açar.
    """
    viz = OusiaVisualizer(graph)
    if save_path:
        viz.save(save_path)
    else:
        viz.show()


# ── Demo (doğrudan çalıştırma) ────────────────────────────────────────────────

if __name__ == "__main__":
    import sys
    from .graph_engine import PatientGraph

    # Örnek graf — graph_engine.py demo verisiyle aynı
    g = PatientGraph()

    g.record_coactivation("work", "boss", 0.9)
    g.record_coactivation("work", "stressed", 0.8)
    g.record_coactivation("boss", "angry", 0.95)
    g.record_coactivation("work", "angry", 0.7)
    g.record_dissonance("work", "fine", 0.85)  # yüksek dissonans
    g.apply_session_decay()

    g.record_coactivation("work", "boss", 0.85)
    g.record_coactivation("mother", "childhood", 0.3)
    g.record_avoidance("childhood", 0.7)
    g.record_avoidance("mother", 0.65)
    g.apply_session_decay()

    g.record_avoidance("mother", 0.72)
    g.record_coactivation("mother", "home", 0.25)
    g.apply_session_decay()

    if len(sys.argv) > 1:
        # python visualizer.py output/my_graph.png
        quick_plot(g, save_path=sys.argv[1])
    else:
        # python visualizer.py  →  interaktif pencere
        # Ortam headless ise PNG kaydeder: output/ousia_demo.png
        import os
        if os.environ.get("DISPLAY") or os.environ.get("TERM_PROGRAM"):
            quick_plot(g)
        else:
            quick_plot(g, save_path="output/ousia_demo.png")
