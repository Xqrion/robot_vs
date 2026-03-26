#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Skill status constants
RUNNING = "RUNNING"
SUCCESS = "SUCCESS"
FAILED = "FAILED"


class BaseSkill(object):
    """Abstract base class for all car skills.

    Each skill represents a single atomic behaviour (e.g. GoTo, Stop).
    Lifecycle:
      1) start()  – called once when the skill becomes active
      2) update() – called every tick; returns RUNNING / SUCCESS / FAILED
      3) stop()   – called once when the skill is cancelled or completed
    """

    def __init__(self, skill_manager):
        self.skill_manager = skill_manager
        self._status = RUNNING

    @property
    def status(self):
        return self._status

    def start(self):
        """Initialise the skill. Called once before the first update()."""
        pass

    def update(self):
        """Execute one step of the skill logic.

        Returns:
            str: RUNNING, SUCCESS, or FAILED
        """
        raise NotImplementedError("Subclasses must implement update()")

    def stop(self):
        """Clean up when the skill is cancelled or has finished."""
        pass
