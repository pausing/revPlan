from datetime import datetime
import json
from typing import Any

class Task:
    def __init__(self, name, assignedTo = None, country = None, project = None, bucket = None, label = None,
                start_date = None, effort = None, due_date = None, priority = 'Normal', is_summary = False):
        self.name = name
        self.assignedTo = assignedTo
        self.country = country
        self.project = project
        self.bucket = bucket
        self.label = label
        self.start_date = start_date
        self.effort = effort
        self.due_date = due_date
        self.priority = priority
        self.is_summary = is_summary
        self.completed = False
        self.subtasks = []
        self._history = []
        self._log_change('Task created')
    
    def _log_change(self, action, field = None, old = None, new = None):
        self._history.append({
            'timestamp': datetime.now(),
            'action': action,
            'field': field,
            'old': old,
            'new': new,
        })

    def add_subtask(self,task):
        if not self.is_summary:
            raise ValueError('Only summary tasks can contain subtasks')
        self.subtasks.append(task)
        self._log_change('Added subtask', field='subtasks', new=task.name)
    
    def mark_complete(self):
        if self.is_summary:
            if all(t.completed for t in self.subtasks):
                self.completed = True
                self._log_change('mark completed')
            else:
                raise ValueError('cannot mark summary task complete until all subtasks are complete')
        else:
            self.completed = True
            self._log_change('marked completed')

    def update(self,field,value):
        if hasattr(self,field):
            old_value = getattr(self,field)
            setattr(self,field,value)
            self._log_change('updated field',field=field,old=old_value,new=value)
        else:
            raise AttributeError('no such field: {}'.format(field))
    
    def is_overdue(self):
        if self.due_date and datetime.now() >self.due_date:
            return True
        return False
    
    def get_history(self):
        return self._history
    
    def __str__(self,indent = 0):
        status = '✓' if self.completed else '✗'
        label = f"{'  '*indent}{status} {self.name} [{self.priority}]"
        if self.due_date:
            label += f" (Due: {self.due_date.strftime('%Y-%m-%d')})"
        lines = [label]
        for sub in self.subtasks:
            lines.append(sub.__str__(indent + 1))
        return '\n'.join(lines)

    def to_dict(self):
        return {
            'name': self.name,
            'assignedTo': self.assignedTo,
            'country': self.country,
            'project': self.project,
            'bucket': self.bucket,
            'label': self.label,
            'start_date': self.start_date.isoformat() if self.start_date else None,
            'effort': self.effort,
            'due_date': self.due_date.isoformat() if self.due_date else None,
            'priority': self.priority,
            'is_summary': self.is_summary,
            'completed': self.completed,
            'subtasks': [t.to_dict() for t in self.subtasks],
            'history': [
                {
                    'timestamp': h['timestamp'].isoformat(),
                    'action': h['action'],
                    'field': h['field'],
                    'old': h['old'],
                    'new': h['new'],
                }
                for h in self._history
            ]
        }

    @classmethod
    def from_dict(cls, data):
        task = cls(
            name=data['name'],
            assignedTo=data.get('assignedTo'),
            country=data.get('country'),
            project=data.get('project'),
            bucket=data.get('bucket'),
            label=data.get('label'),
            start_date=datetime.fromisoformat(data['start_date']) if data.get('start_date') else None,
            effort=data.get('effort'),
            due_date=datetime.fromisoformat(data['due_date']) if data.get('due_date') else None,
            priority=data.get('priority', 'Normal'),
            is_summary=data.get('is_summary', False)
        )
        task.completed = data.get('completed', False)
        task._history = [
            {
                'timestamp': datetime.fromisoformat(h['timestamp']),
                'action': h['action'],
                'field': h['field'],
                'old': h['old'],
                'new': h['new'],
            }
            for h in data.get('history', [])
        ]
        task.subtasks = [cls.from_dict(sub) for sub in data.get('subtasks', [])]
        return task

    def save_to_file(task, filename):
        with open(filename, 'w') as f:
            json.dump(task.to_dict(), f, indent=2)

    def load_from_file(filename):
        with open(filename, 'r') as f:
            data = json.load(f)
        return Task.from_dict(data)
