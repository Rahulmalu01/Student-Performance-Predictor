"""
Academic Recommendation Engine
Generates personalized recommendations based on student profile.
"""


RECOMMENDATIONS_DB = {
    'attendance': {
        'threshold': 75,
        'feature': 'attendance_pct',
        'title': '📅 Improve Attendance',
        'priority': 'High',
        'color': '#FF5252',
        'tips': [
            'Set daily alarms and prepare the night before to avoid missing morning classes.',
            'Join a study group — accountability partners help maintain attendance.',
            'Talk to your professor about any specific barriers to attending class.',
            'Use a habit tracker app to monitor your attendance streak.',
        ],
        'resources': [
            {'name': 'Time Management for Students', 'url': 'https://www.mindtools.com/pages/article/newHTE_00.htm'},
            {'name': 'Building Study Habits', 'url': 'https://www.khanacademy.org/college-careers-more/learnstorm-growth-mindset-activities-us/elementary-and-middle-school-activities/setting-goals/a/setting-goals-for-learning'},
        ],
        'impact': 'Attendance strongly correlates with final grade. Each 10% increase in attendance can improve your CGPA by ~0.3 points.',
    },
    'study_hours': {
        'threshold': 10,
        'feature': 'study_hours_per_week',
        'title': '📚 Increase Study Hours',
        'priority': 'High',
        'color': '#FF6D00',
        'tips': [
            'Use the Pomodoro Technique: 25-minute focused sessions with 5-minute breaks.',
            'Create a fixed weekly study schedule and treat it like a class.',
            'Identify your peak productivity hours and schedule difficult subjects then.',
            'Eliminate distractions — use apps like Forest or Focus@Will.',
        ],
        'resources': [
            {'name': 'Pomodoro Technique Guide', 'url': 'https://francescocirillo.com/pages/pomodoro-technique'},
            {'name': 'Effective Studying Strategies', 'url': 'https://www.coursera.org/articles/study-strategies'},
        ],
        'impact': 'Students studying 15+ hours/week score on average 1.5 grades higher than those studying under 8 hours.',
    },
    'assignments': {
        'threshold': 80,
        'feature': 'assignment_completion_pct',
        'title': '✍️ Complete Assignments Regularly',
        'priority': 'High',
        'color': '#FF4081',
        'tips': [
            'Break large assignments into smaller tasks with mini-deadlines.',
            'Use a planner or Google Calendar to track assignment due dates.',
            'Start assignments immediately when assigned — even just outline them.',
            'Form study groups to collaborate on understanding assignment requirements.',
        ],
        'resources': [
            {'name': 'Assignment Management Tips', 'url': 'https://www.oxfordlearning.com/tips-for-managing-assignments/'},
        ],
        'impact': 'Assignment completion is the 4th most important factor in your final grade. Completing all assignments can boost your score by up to 10%.',
    },
    'midterm': {
        'threshold': 60,
        'feature': 'midterm_score',
        'title': '📝 Strengthen Exam Performance',
        'priority': 'Critical',
        'color': '#D50000',
        'tips': [
            'Review past exam papers and understand the marking scheme.',
            'Practice active recall instead of passive re-reading.',
            'Take mock tests under timed conditions to simulate exam pressure.',
            'Visit office hours to clarify concepts you find difficult.',
            'Join or create a study group focused on exam preparation.',
        ],
        'resources': [
            {'name': 'Active Recall Techniques', 'url': 'https://www.verywellmind.com/active-recall-learning-7368456'},
            {'name': 'How to Prepare for Exams', 'url': 'https://www.coursera.org/articles/how-to-study-for-exams'},
        ],
        'impact': 'Mid-term scores are the strongest predictor of final performance. A 10-point improvement in mid-term often translates to a full grade jump.',
    },
    'stress': {
        'threshold': 7,
        'feature': 'stress_level',
        'title': '🧘 Manage Academic Stress',
        'priority': 'Medium',
        'color': '#7C4DFF',
        'tips': [
            'Practice mindfulness meditation for 10 minutes daily.',
            'Exercise at least 3 times per week to release tension.',
            'Talk to a campus counselor if stress feels overwhelming.',
            'Break your workload into smaller manageable pieces.',
            'Maintain a healthy work-life balance with designated leisure time.',
        ],
        'resources': [
            {'name': 'Headspace Meditation App', 'url': 'https://www.headspace.com/'},
            {'name': 'Stress Management for Students', 'url': 'https://www.verywellmind.com/tips-to-reduce-stress-3145195'},
        ],
        'impact': 'High stress levels reduce cognitive performance by up to 20%. Managing stress effectively can improve focus and retention.',
    },
    'sleep': {
        'threshold': 6,
        'feature': 'sleep_hours_per_night',
        'title': '😴 Improve Sleep Quality',
        'priority': 'Medium',
        'color': '#0091EA',
        'tips': [
            'Aim for 7–9 hours of sleep every night.',
            'Maintain a consistent sleep schedule even on weekends.',
            'Avoid screens 1 hour before bedtime.',
            'Create a relaxing pre-sleep routine (reading, light stretching).',
        ],
        'resources': [
            {'name': 'Sleep and Academic Performance', 'url': 'https://www.sleepfoundation.org/sleep-hygiene/students-and-sleep'},
        ],
        'impact': 'Students sleeping 7+ hours score 10–15% better on average. Sleep is essential for memory consolidation.',
    },
    'participation': {
        'threshold': 5,
        'feature': 'participation_score',
        'title': '🙋 Boost Class Participation',
        'priority': 'Low',
        'color': '#00BCD4',
        'tips': [
            'Prepare 1–2 questions before each class to ask during discussions.',
            'Volunteer to answer questions even if unsure — it reinforces learning.',
            'Sit in the front rows to stay engaged during lectures.',
            'Contribute actively in group discussions and seminars.',
        ],
        'resources': [
            {'name': 'Improving Class Participation', 'url': 'https://www.edutopia.org/article/strategies-student-participation/'},
        ],
        'impact': 'Active participation improves retention by 60% and demonstrates engagement that professors notice.',
    },
    'backlogs': {
        'threshold': 1,
        'feature': 'num_backlogs',
        'title': '⚠️ Clear Academic Backlogs',
        'priority': 'Critical',
        'color': '#BF360C',
        'tips': [
            'Prioritize clearing backlogs before they compound into more failures.',
            'Create a dedicated backlog-clearing study plan each week.',
            'Attend remedial classes or seek peer tutoring for difficult subjects.',
            'Speak with your academic advisor to create a recovery roadmap.',
        ],
        'resources': [
            {'name': 'Academic Recovery Strategies', 'url': 'https://www.edusystem.net/academic-recovery/'},
        ],
        'impact': 'Each uncleared backlog reduces your CGPA by approximately 0.25 points and increases future exam pressure.',
    },
}

PRIORITY_ORDER = {'Critical': 0, 'High': 1, 'Medium': 2, 'Low': 3}


def generate_recommendations(feature_dict):
    """
    Generate sorted, personalized recommendations for a student.

    Args:
        feature_dict: dict of student features

    Returns:
        list of recommendation dicts, sorted by priority
    """
    active_recs = []

    checks = [
        ('attendance',   feature_dict.get('attendance_pct', 100),          lambda v, t: v < t),
        ('study_hours',  feature_dict.get('study_hours_per_week', 15),      lambda v, t: v < t),
        ('assignments',  feature_dict.get('assignment_completion_pct', 100),lambda v, t: v < t),
        ('midterm',      feature_dict.get('midterm_score', 75),             lambda v, t: v < t),
        ('stress',       feature_dict.get('stress_level', 5),               lambda v, t: v > t),
        ('sleep',        feature_dict.get('sleep_hours_per_night', 7),      lambda v, t: v < t),
        ('participation',feature_dict.get('participation_score', 6),        lambda v, t: v < t),
        ('backlogs',     feature_dict.get('num_backlogs', 0),               lambda v, t: v >= t),
    ]

    for key, value, condition in checks:
        rec = RECOMMENDATIONS_DB[key]
        if condition(value, rec['threshold']):
            active_recs.append({
                'key': key,
                'title': rec['title'],
                'priority': rec['priority'],
                'priority_order': PRIORITY_ORDER[rec['priority']],
                'color': rec['color'],
                'tips': rec['tips'],
                'resources': rec['resources'],
                'impact': rec['impact'],
                'current_value': value,
                'target_value': rec['threshold'],
                'feature': rec['feature'],
            })

    # Sort by priority
    active_recs.sort(key=lambda x: x['priority_order'])
    return active_recs


def get_strength_areas(feature_dict):
    """Identify student's strength areas."""
    strengths = []

    if feature_dict.get('attendance_pct', 0) >= 85:
        strengths.append({'area': 'Excellent Attendance', 'icon': '✅', 'value': f"{feature_dict['attendance_pct']}%"})
    if feature_dict.get('study_hours_per_week', 0) >= 15:
        strengths.append({'area': 'Strong Study Habits', 'icon': '📖', 'value': f"{feature_dict['study_hours_per_week']} hrs/week"})
    if feature_dict.get('midterm_score', 0) >= 75:
        strengths.append({'area': 'Strong Exam Performance', 'icon': '🎯', 'value': f"{feature_dict['midterm_score']}/100"})
    if feature_dict.get('assignment_completion_pct', 0) >= 90:
        strengths.append({'area': 'Assignment Completion', 'icon': '✍️', 'value': f"{feature_dict['assignment_completion_pct']}%"})
    if feature_dict.get('stress_level', 10) <= 4:
        strengths.append({'area': 'Well-Managed Stress', 'icon': '🧘', 'value': f"{feature_dict['stress_level']}/10"})
    if feature_dict.get('sleep_hours_per_night', 0) >= 7:
        strengths.append({'area': 'Healthy Sleep Pattern', 'icon': '😴', 'value': f"{feature_dict['sleep_hours_per_night']} hrs/night"})
    if feature_dict.get('num_backlogs', 99) == 0:
        strengths.append({'area': 'No Backlogs', 'icon': '🏆', 'value': 'Clear academic record'})

    return strengths
