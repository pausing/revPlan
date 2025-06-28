import json
from datetime import datetime
from task import Task

class Plan:
    def __init__(self, name):
        self.name = name
        self.created_at = datetime.now()
        self.tasks = []  # list of top-level Task objects

    def add_task(self, task):
        self.tasks.append(task)

    def get_all_tasks(self, include_subtasks=True):
        result = []

        def collect(task):
            result.append(task)
            if include_subtasks:
                for sub in task.subtasks:
                    collect(sub)

        for task in self.tasks:
            collect(task)

        return result

    def get_summary(self):
        all_tasks = self.get_all_tasks()
        total = len(all_tasks)
        done = sum(1 for t in all_tasks if t.completed)
        return {
            'total_tasks': total,
            'completed_tasks': done,
            'completion_rate': round(100 * done / total, 2) if total else 0
        }

    def to_dict(self):
        return {
            'name': self.name,
            'created_at': self.created_at.isoformat(),
            'tasks': [t.to_dict() for t in self.tasks]
        }

    @classmethod
    def from_dict(cls, data):
        plan = cls(data['name'])
        plan.created_at = datetime.fromisoformat(data['created_at'])
        plan.tasks = [Task.from_dict(t) for t in data['tasks']]
        return plan

    def save_to_file(self, filename):
        with open(filename, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)

    @classmethod
    def load_from_file(cls, filename):
        with open(filename, 'r') as f:
            data = json.load(f)
        return cls.from_dict(data)

    def __str__(self):
        lines = [f"Plan: {self.name}", "-"*20]
        for task in self.tasks:
            lines.append(task.__str__())
        return '\n'.join(lines)
