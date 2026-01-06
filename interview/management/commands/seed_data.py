from django.core.management.base import BaseCommand
from interview.models import Topic, Question


class Command(BaseCommand):
    help = 'Seed initial topics and sample questions'

    def handle(self, *args, **options):
        self.stdout.write('Seeding initial data...')

        # Create Topics
        topics_data = [
            {'name': 'Python', 'description': 'Python programming language'},
            {'name': 'SQL', 'description': 'Structured Query Language'},
            {'name': 'DSA', 'description': 'Data Structures and Algorithms'},
            {'name': 'JavaScript', 'description': 'JavaScript programming language'},
            {'name': 'System Design', 'description': 'System design and architecture'},
        ]

        topics = {}
        for topic_data in topics_data:
            topic, created = Topic.objects.get_or_create(
                name=topic_data['name'],
                defaults={'description': topic_data['description']}
            )
            topics[topic_data['name']] = topic
            if created:
                self.stdout.write(self.style.SUCCESS(f'Created topic: {topic.name}'))
            else:
                self.stdout.write(f'Topic already exists: {topic.name}')

        # Create Sample Questions
        questions_data = [
            {
                'topic': 'Python',
                'question_text': 'What is a list comprehension in Python?',
                'ideal_answer': 'A list comprehension is a concise way to create lists in Python. It consists of brackets containing an expression followed by a for clause, then zero or more for or if clauses. For example, [x**2 for x in range(10)] creates a list of squares. List comprehensions are more readable and often faster than equivalent loops.',
                'difficulty': 'EASY'
            },
            {
                'topic': 'Python',
                'question_text': 'Explain the difference between a list and a tuple in Python.',
                'ideal_answer': 'A list is mutable, meaning you can modify it after creation. Lists are defined with square brackets. A tuple is immutable, meaning it cannot be modified after creation. Tuples are defined with parentheses. Lists are generally used for collections of similar items that may change, while tuples are used for fixed collections or as dictionary keys.',
                'difficulty': 'MEDIUM'
            },
            {
                'topic': 'Python',
                'question_text': 'What is the Global Interpreter Lock (GIL) in Python?',
                'ideal_answer': 'The Global Interpreter Lock is a mechanism used in CPython to synchronize access to Python objects, preventing multiple native threads from executing Python bytecodes at once. This means that in CPython, only one thread can execute Python code at a time, even on multi-core systems. The GIL simplifies memory management but can limit performance in CPU-bound multi-threaded programs.',
                'difficulty': 'HARD'
            },
            {
                'topic': 'SQL',
                'question_text': 'What is the difference between INNER JOIN and LEFT JOIN?',
                'ideal_answer': 'INNER JOIN returns only rows that have matching values in both tables. LEFT JOIN returns all rows from the left table and matched rows from the right table. If there is no match, the result is NULL on the right side. LEFT JOIN is useful when you want to include all records from the left table regardless of whether they have matches in the right table.',
                'difficulty': 'MEDIUM'
            },
            {
                'topic': 'SQL',
                'question_text': 'Explain what a database index is and why it is important.',
                'ideal_answer': 'A database index is a data structure that improves the speed of data retrieval operations on a database table. It works like an index in a book, allowing the database to find rows quickly without scanning the entire table. Indexes are created on columns that are frequently used in WHERE clauses or JOIN conditions. While indexes speed up SELECT queries, they can slow down INSERT, UPDATE, and DELETE operations because the index must be maintained.',
                'difficulty': 'MEDIUM'
            },
            {
                'topic': 'DSA',
                'question_text': 'What is the time complexity of binary search?',
                'ideal_answer': 'Binary search has a time complexity of O(log n) in the worst case. This is because with each comparison, the algorithm eliminates half of the remaining elements. Binary search requires the array to be sorted and works by repeatedly dividing the search interval in half.',
                'difficulty': 'EASY'
            },
            {
                'topic': 'DSA',
                'question_text': 'Explain the difference between a stack and a queue.',
                'ideal_answer': 'A stack is a LIFO (Last In First Out) data structure where elements are added and removed from the same end, called the top. Operations are push and pop. A queue is a FIFO (First In First Out) data structure where elements are added at the rear and removed from the front. Operations are enqueue and dequeue. Stacks are used for function calls and undo operations, while queues are used for task scheduling and breadth-first search.',
                'difficulty': 'MEDIUM'
            },
            {
                'topic': 'JavaScript',
                'question_text': 'What is closure in JavaScript?',
                'ideal_answer': 'A closure is a function that has access to variables in its outer lexical scope, even after the outer function has returned. Closures are created when a function is defined inside another function and the inner function references variables from the outer function. This allows for data privacy and function factories. Closures are a fundamental concept in JavaScript that enables many advanced patterns.',
                'difficulty': 'MEDIUM'
            },
            {
                'topic': 'System Design',
                'question_text': 'What is horizontal scaling vs vertical scaling?',
                'ideal_answer': 'Horizontal scaling, also called scaling out, involves adding more machines or servers to your system to handle increased load. Vertical scaling, also called scaling up, involves adding more power to existing machines, such as more CPU, RAM, or storage. Horizontal scaling is generally more cost-effective and provides better fault tolerance, but requires distributed systems design. Vertical scaling is simpler but has hardware limits.',
                'difficulty': 'MEDIUM'
            },
        ]

        created_count = 0
        for q_data in questions_data:
            topic = topics.get(q_data['topic'])
            if not topic:
                self.stdout.write(self.style.WARNING(f'Topic not found: {q_data["topic"]}'))
                continue

            question, created = Question.objects.get_or_create(
                topic=topic,
                question_text=q_data['question_text'],
                defaults={
                    'ideal_answer': q_data['ideal_answer'],
                    'difficulty': q_data['difficulty'],
                    'is_active': True
                }
            )
            if created:
                created_count += 1
                self.stdout.write(self.style.SUCCESS(f'Created question: {question.question_text[:50]}...'))

        self.stdout.write(self.style.SUCCESS(f'\nSeeding complete! Created {created_count} new questions.'))

