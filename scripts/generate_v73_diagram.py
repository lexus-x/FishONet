"""Generate polished, publication-grade v73 Architecture Diagram and Animation with Manim."""
from manim import *


class V73ArchitectureDiagram(Scene):
    def construct(self):
        # Dark Modern Background
        self.camera.background_color = "#0B0F17"

        # Refined Palette
        c_blue = "#38BDF8"
        c_purple = "#A855F7"
        c_emerald = "#10B981"
        c_amber = "#F59E0B"
        c_rose = "#F43F5E"
        c_text = "#F8FAFC"
        c_sub = "#94A3B8"
        c_card = "#161E2E"
        c_feat_bg = "#111827"

        # Title
        title = Text("v73 Multimodal Learned Architecture", font_size=24, color=c_text, weight=BOLD)
        subtitle = Text("Inductive Open-Set Recognition (100% Learning-Based Router)", font_size=13, color=c_sub)
        title_grp = VGroup(title, subtitle).arrange(DOWN, buff=0.08).to_edge(UP, buff=0.25)

        # 1. Input Image Box
        box_input = RoundedRectangle(corner_radius=0.12, height=0.60, width=2.4, color=c_blue, fill_color=c_card, fill_opacity=0.95, stroke_width=2)
        txt_input = Text("Input Image x", font_size=13, color=c_blue, weight=BOLD)
        grp_input = VGroup(box_input, txt_input).move_to(UP * 2.3)

        # 2. Encoder Box
        box_enc = RoundedRectangle(corner_radius=0.12, height=0.62, width=3.6, color=c_purple, fill_color=c_card, fill_opacity=0.95, stroke_width=2)
        txt_enc = Text("BioCLIP-2.5 (ViT-H/14)", font_size=13, color=c_purple, weight=BOLD)
        grp_enc = VGroup(box_enc, txt_enc).next_to(grp_input, DOWN, buff=0.35)

        arr_input_enc = Arrow(grp_input.get_bottom(), grp_enc.get_top(), buff=0.06, color=c_blue, stroke_width=2.5, max_tip_length_to_length_ratio=0.22)
        lbl_z = Text("Embedding z ∈ R^1024", font_size=11, color=c_sub).next_to(grp_enc, DOWN, buff=0.08)

        # 3. Middle Tier: Left (Seen), Center (Features), Right (Unseen)
        # Center: 13-D Features
        box_feat = RoundedRectangle(corner_radius=0.12, height=1.05, width=2.7, color="#374151", fill_color=c_feat_bg, fill_opacity=0.95, stroke_width=1.5)
        t_f1 = Text("13-D Features Φ(x)", font_size=11, color=c_text, weight=BOLD)
        t_f2 = Text("• Prototypes Sim\n• Shannon Entropy -H(p)\n• Contrast Margins\n• Likelihood Ratios", font_size=8.5, color=c_sub, line_spacing=0.85)
        grp_feat_txt = VGroup(t_f1, t_f2).arrange(DOWN, buff=0.06)
        grp_feat = VGroup(box_feat, grp_feat_txt).move_to(DOWN * 0.15)

        # Left: Seen Head
        box_seen = RoundedRectangle(corner_radius=0.12, height=1.05, width=3.3, color=c_emerald, fill_color=c_card, fill_opacity=0.95, stroke_width=2)
        t_s1 = Text("Seen Candidate Head", font_size=12, color=c_emerald, weight=BOLD)
        t_s2 = Text("5,795 Train Prototypes P_s", font_size=10, color=c_sub)
        t_s3 = Text("c* = argmax cos(z, P_s)", font_size=10.5, color=c_emerald)
        grp_seen_txt = VGroup(t_s1, t_s2, t_s3).arrange(DOWN, buff=0.08)
        grp_seen = VGroup(box_seen, grp_seen_txt).move_to(LEFT * 3.3 + DOWN * 0.15)

        # Right: Multimodal Unseen Head
        box_uns = RoundedRectangle(corner_radius=0.12, height=1.05, width=3.5, color=c_amber, fill_color=c_card, fill_opacity=0.95, stroke_width=2)
        t_u1 = Text("Multimodal Unseen Head", font_size=12, color=c_amber, weight=BOLD)
        t_u2 = Text("11,598 Anchors (Text + iNat/ToL)", font_size=10, color=c_sub)
        t_u3 = Text("u* = argmax cos(z, E_u)", font_size=10.5, color=c_amber)
        grp_uns_txt = VGroup(t_u1, t_u2, t_u3).arrange(DOWN, buff=0.08)
        grp_uns = VGroup(box_uns, grp_uns_txt).move_to(RIGHT * 3.3 + DOWN * 0.15)

        # Clean branching lines from z to heads & features
        line_split_y = grp_enc.get_bottom()[1] - 0.28
        p_split = np.array([0, line_split_y, 0])
        p_seen_top = np.array([grp_seen.get_top()[0], line_split_y, 0])
        p_uns_top = np.array([grp_uns.get_top()[0], line_split_y, 0])

        arr_to_feat = Arrow(p_split, grp_feat.get_top(), buff=0.06, color=c_purple, stroke_width=2, max_tip_length_to_length_ratio=0.2)
        arr_to_seen = Arrow(p_seen_top, grp_seen.get_top(), buff=0.06, color=c_emerald, stroke_width=2, max_tip_length_to_length_ratio=0.2)
        arr_to_uns = Arrow(p_uns_top, grp_uns.get_top(), buff=0.06, color=c_amber, stroke_width=2, max_tip_length_to_length_ratio=0.2)
        wire_h = Line(p_seen_top, p_uns_top, color="#475569", stroke_width=1.5)

        # 4. Learned Soft Gate Box
        box_gate = RoundedRectangle(corner_radius=0.12, height=0.88, width=4.2, color=c_rose, fill_color=c_card, fill_opacity=0.95, stroke_width=2.5)
        t_g1 = Text("Learned Logistic Soft Gate", font_size=13, color=c_rose, weight=BOLD)
        t_g2 = Text("P(seen | x) = σ( W·Φ(x) + b )", font_size=12, color=c_text)
        t_g3 = Text("MLE Optimized on Validation Set (Inductive, O(1))", font_size=9, color=c_sub)
        grp_gate_txt = VGroup(t_g1, t_g2, t_g3).arrange(DOWN, buff=0.06)
        grp_gate = VGroup(box_gate, grp_gate_txt).next_to(grp_feat, DOWN, buff=0.40)

        arr_feat_gate = Arrow(grp_feat.get_bottom(), grp_gate.get_top(), buff=0.06, color=c_rose, stroke_width=2.2, max_tip_length_to_length_ratio=0.22)

        # 5. Decision Box & Output
        box_dec = RoundedRectangle(corner_radius=0.1, height=0.52, width=2.6, color=c_blue, fill_color=c_card, fill_opacity=0.95, stroke_width=2)
        t_dec = Text("P(seen | x) ≥ 0.50 ?", font_size=11.5, color=c_blue, weight=BOLD)
        grp_dec = VGroup(box_dec, t_dec).next_to(grp_gate, DOWN, buff=0.35)

        arr_gate_dec = Arrow(grp_gate.get_bottom(), grp_dec.get_top(), buff=0.06, color=c_blue, stroke_width=2.2, max_tip_length_to_length_ratio=0.22)

        # Output boxes
        box_out_s = RoundedRectangle(corner_radius=0.1, height=0.50, width=2.5, color=c_emerald, fill_color=c_card, fill_opacity=0.95, stroke_width=2)
        txt_out_s = Text("Predict c* (Seen)", font_size=11, color=c_emerald, weight=BOLD)
        grp_out_s = VGroup(box_out_s, txt_out_s).move_to(LEFT * 3.0 + DOWN * 3.1)

        box_out_u = RoundedRectangle(corner_radius=0.1, height=0.50, width=2.5, color=c_amber, fill_color=c_card, fill_opacity=0.95, stroke_width=2)
        txt_out_u = Text("Predict u* (Novel)", font_size=11, color=c_amber, weight=BOLD)
        grp_out_u = VGroup(box_out_u, txt_out_u).move_to(RIGHT * 3.0 + DOWN * 3.1)

        arr_yes = Arrow(grp_dec.get_left(), grp_out_s.get_top(), buff=0.06, color=c_emerald, stroke_width=2, max_tip_length_to_length_ratio=0.2)
        lbl_yes = Text("YES (Seen)", font_size=9.5, color=c_emerald, weight=BOLD).move_to(LEFT * 2.3 + DOWN * 2.5)

        arr_no = Arrow(grp_dec.get_right(), grp_out_u.get_top(), buff=0.06, color=c_amber, stroke_width=2, max_tip_length_to_length_ratio=0.2)
        lbl_no = Text("NO (Unseen)", font_size=9.5, color=c_amber, weight=BOLD).move_to(RIGHT * 2.3 + DOWN * 2.5)

        # Add all elements
        self.add(
            title_grp,
            grp_input, arr_input_enc, grp_enc, lbl_z,
            wire_h, arr_to_feat, arr_to_seen, arr_to_uns,
            grp_seen, grp_uns, grp_feat,
            arr_feat_gate, grp_gate,
            arr_gate_dec, grp_dec,
            arr_yes, lbl_yes, grp_out_s,
            arr_no, lbl_no, grp_out_u
        )


class V73ArchitectureAnimation(Scene):
    def construct(self):
        self.camera.background_color = "#0B0F17"

        # Refined Palette
        c_blue = "#38BDF8"
        c_purple = "#A855F7"
        c_emerald = "#10B981"
        c_amber = "#F59E0B"
        c_rose = "#F43F5E"
        c_text = "#F8FAFC"
        c_sub = "#94A3B8"
        c_card = "#161E2E"
        c_feat_bg = "#111827"

        # Title
        title = Text("v73 Multimodal Learned Architecture", font_size=24, color=c_text, weight=BOLD)
        subtitle = Text("Inductive Open-Set Recognition (100% Learning-Based Router)", font_size=13, color=c_sub)
        title_grp = VGroup(title, subtitle).arrange(DOWN, buff=0.08).to_edge(UP, buff=0.25)

        # 1. Input Image Box
        box_input = RoundedRectangle(corner_radius=0.12, height=0.60, width=2.4, color=c_blue, fill_color=c_card, fill_opacity=0.95, stroke_width=2)
        txt_input = Text("Input Image x", font_size=13, color=c_blue, weight=BOLD)
        grp_input = VGroup(box_input, txt_input).move_to(UP * 2.3)

        # 2. Encoder Box
        box_enc = RoundedRectangle(corner_radius=0.12, height=0.62, width=3.6, color=c_purple, fill_color=c_card, fill_opacity=0.95, stroke_width=2)
        txt_enc = Text("BioCLIP-2.5 (ViT-H/14)", font_size=13, color=c_purple, weight=BOLD)
        grp_enc = VGroup(box_enc, txt_enc).next_to(grp_input, DOWN, buff=0.35)

        arr_input_enc = Arrow(grp_input.get_bottom(), grp_enc.get_top(), buff=0.06, color=c_blue, stroke_width=2.5, max_tip_length_to_length_ratio=0.22)
        lbl_z = Text("Embedding z ∈ R^1024", font_size=11, color=c_sub).next_to(grp_enc, DOWN, buff=0.08)

        # 3. Middle Tier: Left (Seen), Center (Features), Right (Unseen)
        box_feat = RoundedRectangle(corner_radius=0.12, height=1.05, width=2.7, color="#374151", fill_color=c_feat_bg, fill_opacity=0.95, stroke_width=1.5)
        t_f1 = Text("13-D Features Φ(x)", font_size=11, color=c_text, weight=BOLD)
        t_f2 = Text("• Prototypes Sim\n• Shannon Entropy -H(p)\n• Contrast Margins\n• Likelihood Ratios", font_size=8.5, color=c_sub, line_spacing=0.85)
        grp_feat_txt = VGroup(t_f1, t_f2).arrange(DOWN, buff=0.06)
        grp_feat = VGroup(box_feat, grp_feat_txt).move_to(DOWN * 0.15)

        box_seen = RoundedRectangle(corner_radius=0.12, height=1.05, width=3.3, color=c_emerald, fill_color=c_card, fill_opacity=0.95, stroke_width=2)
        t_s1 = Text("Seen Candidate Head", font_size=12, color=c_emerald, weight=BOLD)
        t_s2 = Text("5,795 Train Prototypes P_s", font_size=10, color=c_sub)
        t_s3 = Text("c* = argmax cos(z, P_s)", font_size=10.5, color=c_emerald)
        grp_seen_txt = VGroup(t_s1, t_s2, t_s3).arrange(DOWN, buff=0.08)
        grp_seen = VGroup(box_seen, grp_seen_txt).move_to(LEFT * 3.3 + DOWN * 0.15)

        box_uns = RoundedRectangle(corner_radius=0.12, height=1.05, width=3.5, color=c_amber, fill_color=c_card, fill_opacity=0.95, stroke_width=2)
        t_u1 = Text("Multimodal Unseen Head", font_size=12, color=c_amber, weight=BOLD)
        t_u2 = Text("11,598 Anchors (Text + iNat/ToL)", font_size=10, color=c_sub)
        t_u3 = Text("u* = argmax cos(z, E_u)", font_size=10.5, color=c_amber)
        grp_uns_txt = VGroup(t_u1, t_u2, t_u3).arrange(DOWN, buff=0.08)
        grp_uns = VGroup(box_uns, grp_uns_txt).move_to(RIGHT * 3.3 + DOWN * 0.15)

        line_split_y = grp_enc.get_bottom()[1] - 0.28
        p_split = np.array([0, line_split_y, 0])
        p_seen_top = np.array([grp_seen.get_top()[0], line_split_y, 0])
        p_uns_top = np.array([grp_uns.get_top()[0], line_split_y, 0])

        arr_to_feat = Arrow(p_split, grp_feat.get_top(), buff=0.06, color=c_purple, stroke_width=2, max_tip_length_to_length_ratio=0.2)
        arr_to_seen = Arrow(p_seen_top, grp_seen.get_top(), buff=0.06, color=c_emerald, stroke_width=2, max_tip_length_to_length_ratio=0.2)
        arr_to_uns = Arrow(p_uns_top, grp_uns.get_top(), buff=0.06, color=c_amber, stroke_width=2, max_tip_length_to_length_ratio=0.2)
        wire_h = Line(p_seen_top, p_uns_top, color="#475569", stroke_width=1.5)

        box_gate = RoundedRectangle(corner_radius=0.12, height=0.88, width=4.2, color=c_rose, fill_color=c_card, fill_opacity=0.95, stroke_width=2.5)
        t_g1 = Text("Learned Logistic Soft Gate", font_size=13, color=c_rose, weight=BOLD)
        t_g2 = Text("P(seen | x) = σ( W·Φ(x) + b )", font_size=12, color=c_text)
        t_g3 = Text("MLE Optimized on Validation Set (Inductive, O(1))", font_size=9, color=c_sub)
        grp_gate_txt = VGroup(t_g1, t_g2, t_g3).arrange(DOWN, buff=0.06)
        grp_gate = VGroup(box_gate, grp_gate_txt).next_to(grp_feat, DOWN, buff=0.40)

        arr_feat_gate = Arrow(grp_feat.get_bottom(), grp_gate.get_top(), buff=0.06, color=c_rose, stroke_width=2.2, max_tip_length_to_length_ratio=0.22)

        box_dec = RoundedRectangle(corner_radius=0.1, height=0.52, width=2.6, color=c_blue, fill_color=c_card, fill_opacity=0.95, stroke_width=2)
        t_dec = Text("P(seen | x) ≥ 0.50 ?", font_size=11.5, color=c_blue, weight=BOLD)
        grp_dec = VGroup(box_dec, t_dec).next_to(grp_gate, DOWN, buff=0.35)

        arr_gate_dec = Arrow(grp_gate.get_bottom(), grp_dec.get_top(), buff=0.06, color=c_blue, stroke_width=2.2, max_tip_length_to_length_ratio=0.22)

        box_out_s = RoundedRectangle(corner_radius=0.1, height=0.50, width=2.5, color=c_emerald, fill_color=c_card, fill_opacity=0.95, stroke_width=2)
        txt_out_s = Text("Predict c* (Seen)", font_size=11, color=c_emerald, weight=BOLD)
        grp_out_s = VGroup(box_out_s, txt_out_s).move_to(LEFT * 3.0 + DOWN * 3.1)

        box_out_u = RoundedRectangle(corner_radius=0.1, height=0.50, width=2.5, color=c_amber, fill_color=c_card, fill_opacity=0.95, stroke_width=2)
        txt_out_u = Text("Predict u* (Novel)", font_size=11, color=c_amber, weight=BOLD)
        grp_out_u = VGroup(box_out_u, txt_out_u).move_to(RIGHT * 3.0 + DOWN * 3.1)

        arr_yes = Arrow(grp_dec.get_left(), grp_out_s.get_top(), buff=0.06, color=c_emerald, stroke_width=2, max_tip_length_to_length_ratio=0.2)
        lbl_yes = Text("YES (Seen)", font_size=9.5, color=c_emerald, weight=BOLD).move_to(LEFT * 2.3 + DOWN * 2.5)

        arr_no = Arrow(grp_dec.get_right(), grp_out_u.get_top(), buff=0.06, color=c_amber, stroke_width=2, max_tip_length_to_length_ratio=0.2)
        lbl_no = Text("NO (Unseen)", font_size=9.5, color=c_amber, weight=BOLD).move_to(RIGHT * 2.3 + DOWN * 2.5)

        # Animation Sequence
        self.play(FadeIn(title_grp, shift=UP * 0.2), run_time=0.6)
        self.play(FadeIn(grp_input), GrowArrow(arr_input_enc), FadeIn(grp_enc), FadeIn(lbl_z), run_time=0.8)
        self.play(
            Create(wire_h),
            GrowArrow(arr_to_seen), FadeIn(grp_seen),
            GrowArrow(arr_to_uns), FadeIn(grp_uns),
            GrowArrow(arr_to_feat), FadeIn(grp_feat),
            run_time=0.9
        )
        self.play(GrowArrow(arr_feat_gate), FadeIn(grp_gate), run_time=0.7)
        self.play(GrowArrow(arr_gate_dec), FadeIn(grp_dec), run_time=0.5)
        self.play(
            GrowArrow(arr_yes), FadeIn(lbl_yes), FadeIn(grp_out_s),
            GrowArrow(arr_no), FadeIn(lbl_no), FadeIn(grp_out_u),
            run_time=0.8
        )
        self.wait(1.5)


if __name__ == '__main__':
    pass
