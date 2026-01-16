# data_access.py
from sqlalchemy.orm import Session
from database import Session as SessionModel
import json

class SessionRepository:
    def __init__(self, db: Session):
        self.db = db
    
    def create_session(self, session_data: dict):
        db_session = SessionModel(
            name=session_data["name"],
            service_no=session_data["service_no"],
            service_year=session_data["service_year"],
            timestamp=session_data.get("timestamp", None),
            hits=session_data["hits"],
            group_size=session_data["group_size"],
            holes_data=json.dumps(session_data["holes"]),
            image_path=session_data.get("image_path", "")
        )
        self.db.add(db_session)
        self.db.commit()
        self.db.refresh(db_session)
        return db_session
    
    def get_all_sessions(self):
        sessions = self.db.query(SessionModel).all()
        return [
            {
                "id": s.id,
                "name": s.name,
                "service_no": s.service_no,
                "service_year": s.service_year,
                "timestamp": s.timestamp.isoformat() if s.timestamp else None,
                "hits": s.hits,
                "group_size": s.group_size,
                "holes": json.loads(s.holes_data) if s.holes_data else [],
                "image_path": s.image_path
            }
            for s in sessions
        ]
    
    def get_session_by_id(self, session_id: int):
        session = self.db.query(SessionModel).filter(SessionModel.id == session_id).first()
        if session:
            return {
                "id": session.id,
                "name": session.name,
                "service_no": session.service_no,
                "service_year": session.service_year,
                "timestamp": session.timestamp.isoformat() if session.timestamp else None,
                "hits": session.hits,
                "group_size": session.group_size,
                "holes": json.loads(session.holes_data) if session.holes_data else [],
                "image_path": session.image_path
            }
        return None