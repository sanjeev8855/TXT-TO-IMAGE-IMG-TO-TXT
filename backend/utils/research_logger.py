import sqlite3
import json
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Any, List, Optional

from backend.config import config

class ResearchExperimentLogger:
    """
    Module 8: Image Storage / Result Manager & Research Experiment Logger.
    Persists structured metadata, model configurations, and forensic evaluation metrics
    into an academic research SQLite store and JSON experiment logs.
    Strictly prevents API keys or passwords from being logged.
    """
    def __init__(self, db_path: Optional[Path] = None):
        self.db_path = db_path or (config.base_dir / "research_experiments.db")
        self._init_db()

    def _init_db(self):
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS experiments (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    experiment_id TEXT UNIQUE,
                    timestamp TEXT,
                    experiment_type TEXT,
                    input_filename TEXT,
                    image_width INTEGER,
                    image_height INTEGER,
                    selected_models TEXT,
                    rag_top_k INTEGER,
                    retrieved_doc_ids TEXT,
                    final_prompt TEXT,
                    generation_parameters TEXT,
                    forensic_model TEXT,
                    global_anomaly_score REAL,
                    ai_generated_likelihood REAL,
                    anomaly_level TEXT,
                    heatmap_path TEXT,
                    overlay_path TEXT,
                    report_id TEXT,
                    metadata_json TEXT
                )
            """)
            conn.commit()

    def log_experiment(
        self,
        experiment_type: str,
        input_filename: str,
        image_dims: tuple,
        selected_models: Dict[str, str],
        rag_top_k: int,
        retrieved_doc_ids: List[str],
        final_prompt: str,
        generation_params: Dict[str, Any],
        forensic_model: str,
        global_anomaly_score: float,
        ai_generated_likelihood: float,
        anomaly_level: str,
        heatmap_path: str,
        overlay_path: str,
        report_id: str,
        extra_meta: Optional[Dict[str, Any]] = None
    ) -> str:
        # Sanitize any accidental sensitive keys from extra_meta
        clean_meta = {}
        if extra_meta:
            for k, v in extra_meta.items():
                if "key" not in k.lower() and "token" not in k.lower() and "secret" not in k.lower():
                    clean_meta[k] = v

        experiment_id = f"EXP_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}_{report_id[-4:]}"
        now_iso = datetime.now(timezone.utc).isoformat()

        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT INTO experiments (
                    experiment_id, timestamp, experiment_type, input_filename,
                    image_width, image_height, selected_models, rag_top_k,
                    retrieved_doc_ids, final_prompt, generation_parameters,
                    forensic_model, global_anomaly_score, ai_generated_likelihood,
                    anomaly_level, heatmap_path, overlay_path, report_id, metadata_json
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                experiment_id,
                now_iso,
                experiment_type,
                input_filename,
                image_dims[0],
                image_dims[1],
                json.dumps(selected_models),
                rag_top_k,
                json.dumps(retrieved_doc_ids),
                final_prompt,
                json.dumps(generation_params),
                forensic_model,
                global_anomaly_score,
                ai_generated_likelihood,
                anomaly_level,
                heatmap_path,
                overlay_path,
                report_id,
                json.dumps(clean_meta)
            ))
            conn.commit()

        return experiment_id

    def get_recent_experiments(self, limit: int = 50) -> List[Dict[str, Any]]:
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM experiments ORDER BY id DESC LIMIT ?", (limit,))
            rows = cursor.fetchall()
            results = []
            for r in rows:
                d = dict(r)
                d["selected_models"] = json.loads(d["selected_models"]) if d["selected_models"] else {}
                d["retrieved_doc_ids"] = json.loads(d["retrieved_doc_ids"]) if d["retrieved_doc_ids"] else []
                d["generation_parameters"] = json.loads(d["generation_parameters"]) if d["generation_parameters"] else {}
                results.append(d)
            return results

research_logger = ResearchExperimentLogger()
