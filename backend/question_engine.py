"""
question_engine.py
Drives the hierarchical, adaptive question flow for AyurGenX.
"""

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


class QuestionEngine:
    """
    State-machine that selects the next question based on:
      • detected CV conditions (from the face scan)
      • user-selected primary concerns
      • previous answers
    At most ~5 questions are ever shown to the user.
    """

    MAX_QUESTIONS = 5

    def __init__(self, tree_path: str):
        with open(tree_path, encoding="utf-8") as f:
            self.tree: Dict = json.load(f)

    # ── public API ────────────────────────────────────────────────────────────

    def get_initial_questions(self) -> List[Dict]:
        return self.tree["initial_questions"]

    def get_next_question(
        self,
        cv_conditions: List[str],
        primary_concerns: List[str],
        answers: Dict[str, Any],
        question_count: int,
    ) -> Optional[Dict]:
        """
        Return the next question dict, or None if the assessment is complete.
        """
        if question_count >= self.MAX_QUESTIONS:
            return None

        last_answer = answers.get("_last_answer", {})
        next_id     = last_answer.get("next") if last_answer else None

        # ── 1. follow the tree pointer if we have one ─────────────────────
        if next_id and next_id != "END":
            q = self._find_by_id(next_id)
            if q:
                return q

        if next_id == "END":
            return None

        # ── 2. condition-driven branch entry ──────────────────────────────
        already_asked = set(answers.keys()) | {"_last_answer"}
        for condition in self._priority_sorted(cv_conditions):
            branch = self.tree["condition_branches"].get(condition)
            if not branch:
                continue
            entry = branch["entry_question"]
            if entry["id"] not in already_asked:
                return entry

        # ── 3. primary-concern driven question ────────────────────────────
        for concern in primary_concerns:
            cq = self.tree["concern_questions"].get(concern)
            if cq and cq["id"] not in already_asked:
                return cq

        # ── 4. shared lifestyle questions as fallback ─────────────────────
        for q in self.tree["shared_questions"].values():
            if q["id"] not in already_asked:
                return q

        return None  # all done

    def generate_summary(
        self,
        cv_results: List[Dict],
        cv_extra: Dict,
        answers: Dict[str, Any],
        primary_concerns: List[str],
        age_group: str,
    ) -> Dict:
        """
        Synthesise all collected data into a structured health summary.
        Returns a dict consumed by the frontend results page.
        """
        # Dosha tally
        dosha_counts: Dict[str, int] = {}
        for r in cv_results:
            d = r.get("ayurvedic_dosha", "Tridosha")
            dosha_counts[d] = dosha_counts.get(d, 0) + 1

        dominant_dosha = (
            max(dosha_counts, key=dosha_counts.get)
            if dosha_counts else "Tridosha"
        )

        conditions    = [r["condition"]  for r in cv_results]
        severities    = {r["condition"]: r["severity"] for r in cv_results}
        skin_tone     = cv_extra.get("skin_tone", "Medium")
        glow_score    = cv_extra.get("glow_score", 0.5)
        dullness      = cv_extra.get("dullness", 0.5)
        redness_idx   = cv_extra.get("redness_index", 0.0)
        roughness     = cv_extra.get("roughness", 0.3)

        recommendations = self._build_recommendations(
            conditions, answers, primary_concerns, dominant_dosha,
            glow_score, roughness, age_group
        )

        return {
            "dominant_dosha":    dominant_dosha,
            "dosha_breakdown":   dosha_counts,
            "detected_conditions": cv_results,
            "skin_metrics": {
                "skin_tone":    skin_tone,
                "glow_score":   round(glow_score * 100),
                "dullness_pct": round(dullness  * 100),
                "redness_index": round(redness_idx * 100, 1),
                "roughness_pct": round(roughness  * 100),
            },
            "recommendations":   recommendations,
            "lifestyle_flags":   self._lifestyle_flags(answers),
        }

    # ── private ───────────────────────────────────────────────────────────────

    def _find_by_id(self, qid: str) -> Optional[Dict]:
        """Search all question stores for a question by id."""
        # shared
        sq = self.tree["shared_questions"].get(qid)
        if sq: return sq
        # concern
        for cq in self.tree["concern_questions"].values():
            if cq["id"] == qid: return cq
        # follow-ups inside condition branches
        for branch in self.tree["condition_branches"].values():
            for fq in branch.get("follow_ups", {}).values():
                if fq["id"] == qid: return fq
        return None

    def _priority_sorted(self, conditions: List[str]) -> List[str]:
        """Sort conditions by their configured priority (lower = first)."""
        def prio(c):
            b = self.tree["condition_branches"].get(c)
            return b["priority"] if b else 99
        return sorted(conditions, key=prio)

    @staticmethod
    def _lifestyle_flags(answers: Dict) -> List[str]:
        flags = []
        water  = answers.get("lifestyle_water")
        sleep  = answers.get("lifestyle_sleep")
        stress = answers.get("lifestyle_stress")

        if water  == "low":    flags.append("Low hydration")
        if sleep  == "bad":    flags.append("Poor sleep quality")
        if stress in ("work","personal","health","anxiety"):
            flags.append("Elevated stress levels")
        return flags

    @staticmethod
    def _build_recommendations(
        conditions, answers, concerns, dosha, glow, roughness, age
    ) -> List[Dict]:
        recs = []

        # ── dosha-based base rec ──────────────────────────────────────────
        dosha_recs = {
            "Pitta":   ("Cool & Calm Routine 🌙",
                        "Avoid spicy food, direct sun. Use aloe vera, rose water, and cooling herbs like Amla."),
            "Kapha":   ("Detox & Energise 🌿",
                        "Use warming herbs like Triphala, dry-brush skin, exercise regularly, reduce dairy."),
            "Vata":    ("Nourish & Hydrate 🌊",
                        "Use sesame oil for abhyanga (self-massage), stay warm, increase healthy fats."),
            "Tridosha":("Balanced Lifestyle ✨",
                        "Follow a consistent routine, seasonal eating, yoga, and stress management."),
        }
        title, desc = dosha_recs.get(dosha, dosha_recs["Tridosha"])
        recs.append({"icon": "🏵️", "title": f"Ayurvedic Type: {dosha}", "desc": desc})

        # ── condition-specific recs ───────────────────────────────────────
        if any(c in conditions for c in ["acne","inflammatory acne"]):
            recs.append({
                "icon": "🌸",
                "title": "Acne Care",
                "desc": "Neem face wash, avoid touching face, reduce dairy/sugar intake. Turmeric-honey mask twice a week."
            })
        if "dark spots" in conditions or "pigmentation" in conditions:
            recs.append({
                "icon": "☀️",
                "title": "Pigmentation Defence",
                "desc": "Apply SPF 50 daily. Vitamin C serum morning, Kojic acid night cream. Papaya + lemon DIY mask."
            })
        if "wrinkles" in conditions:
            recs.append({
                "icon": "💧",
                "title": "Anti-Ageing Hydration",
                "desc": "Hyaluronic acid serum, Ashwagandha internally, 8+ hrs sleep. Avoid smoking & excessive sun."
            })
        if "Redness" in conditions:
            recs.append({
                "icon": "🧊",
                "title": "Redness Soothing",
                "desc": "Aloe vera gel directly applied, fragrance-free products, reduce alcohol & spicy food."
            })

        # ── lifestyle flags ───────────────────────────────────────────────
        if answers.get("lifestyle_water") == "low":
            recs.append({
                "icon": "🚰",
                "title": "Hydrate More",
                "desc": "Aim for 8 glasses/day. Add cucumber or mint for taste. Hydration is the #1 skin fix."
            })
        if answers.get("lifestyle_sleep") == "bad":
            recs.append({
                "icon": "🌙",
                "title": "Sleep Reset",
                "desc": "Brahmi oil scalp massage before bed, avoid screens 1 hr before sleep, warm milk with nutmeg."
            })

        # ── glow score ────────────────────────────────────────────────────
        if glow < 0.45:
            recs.append({
                "icon": "✨",
                "title": "Boost Your Glow",
                "desc": "Your skin looks a little dull. Try: Vitamin C-rich foods, facial yoga, and a 10-min morning walk."
            })

        return recs
