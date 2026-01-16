# services.py
from backend import ShootingAnalyzer
from data_access import SessionRepository
from database import SessionLocal
from datetime import datetime

class ShootingService:
    def __init__(self):
        self.analyzer = ShootingAnalyzer()
        self.current_session_index = -1
        self.all_sessions = []
    
    def get_session_data(self):
        """Return current session information for web interface"""
        return {
            'current_session_index': self.analyzer.current_session_index,
            'total_sessions': len(self.analyzer.all_sessions),
            'active_hits': len(self.analyzer.all_sessions[self.analyzer.current_session_index]['holes']) if self.analyzer.current_session_index != -1 else 0,
            'all_sessions': [
                {
                    'name': session.get('name', 'N/A'),
                    'service_no': session.get('service_no', 'N/A'),
                    'service_year': session.get('service_year', 'N/A'),
                    'hits': len(session.get('holes', [])),
                    'group_size': session.get('group_size', 0.0),
                    'timestamp': session.get('timestamp', 'N/A')
                }
                for session in self.analyzer.all_sessions
            ]
        }
    
    def start_session(self, name: str, service_no: str, service_year: str):
        """Start a new session and save to database"""
        # Use the original backend's start_session method
        self.analyzer.start_session(name, service_no, service_year)
        
        # Save to database
        db = SessionLocal()
        repo = SessionRepository(db)
        repo.create_session({
            "name": name,
            "service_no": service_no,
            "service_year": service_year,
            "timestamp": datetime.now(),
            "hits": 0,
            "group_size": 0.0,
            "holes": [],
            "image_path": ""
        })
        db.close()
        
        # Ensure session index is set correctly
        self.current_session_index = len(self.analyzer.all_sessions) - 1
        
        return {"success": True}

    def stop_session(self):
        """Stop current session and save to database"""
        result = self.analyzer.stop_session()
        
        if result["success"]:
            # Update database
            db = SessionLocal()
            repo = SessionRepository(db)
            # Get the last session from analyzer
            if len(self.analyzer.all_sessions) > 0:
                last_session = self.analyzer.all_sessions[-1]
                # Update in database
                from database import Session as SessionModel
                db_obj = db.query(SessionModel).filter(SessionModel.id == len(self.analyzer.all_sessions)).first()
                if db_obj:
                    db_obj.hits = result["hits"]
                    db_obj.group_size = result["group_size"]
                    db.commit()
            db.close()
        
        # Reset session index
        self.current_session_index = -1
        
        return result
    
    def get_all_sessions(self):
        """Get all sessions from database"""
        db = SessionLocal()
        repo = SessionRepository(db)
        sessions = repo.get_all_sessions()
        db.close()
        return sessions
    
    def process_frame(self):
        """Process frame for detection"""
        return self.analyzer.process_frame()
    
    def get_display_frame(self):
        """Get display frame"""
        return self.analyzer.get_display_frame()
    
    def _grab_frame(self):
        """Grab frame from camera"""
        return self.analyzer._grab_frame()