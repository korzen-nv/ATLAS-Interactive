import uuid
import time
from dataclasses import dataclass, field
from typing import Dict, Optional


@dataclass
class Session:
    session_id: str
    controller: object  # WebController instance
    created_at: float = field(default_factory=time.time)
    last_active: float = field(default_factory=time.time)

    def touch(self):
        self.last_active = time.time()


class SessionManager:
    def __init__(self):
        self._sessions: Dict[str, Session] = {}

    def create_session(self, controller) -> str:
        session_id = uuid.uuid4().hex[:12]
        self._sessions[session_id] = Session(
            session_id=session_id,
            controller=controller,
        )
        return session_id

    def get_session(self, session_id: str) -> Optional[Session]:
        session = self._sessions.get(session_id)
        if session:
            session.touch()
        return session

    def destroy_session(self, session_id: str) -> bool:
        if session_id in self._sessions:
            del self._sessions[session_id]
            return True
        return False

    def list_sessions(self) -> list:
        return [
            {"session_id": s.session_id, "created_at": s.created_at, "last_active": s.last_active}
            for s in self._sessions.values()
        ]

    @property
    def active_session(self) -> Optional[Session]:
        """Get the most recently active session (convenience for single-user)."""
        if not self._sessions:
            return None
        return max(self._sessions.values(), key=lambda s: s.last_active)
