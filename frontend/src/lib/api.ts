// API service for Django backend
const API_URL = import.meta.env.VITE_API_URL || (import.meta.env.MODE === 'development' ? 'http://localhost:8000/api' : 'https://ai-interviewer-backend-f9t5.onrender.com/api');

interface UserProfile {
  id: number;
  username: string;
  email: string;
  name: string | null;
  is_active: boolean;
  role: 'ADMIN' | 'USER';
  access_type: 'TRIAL' | 'FULL' | null; // Access type is mostly relevant for USER
  has_used_trial: boolean;
  plain_password?: string;
  created_at: string;
  updated_at: string;
}

interface Course {
  id: number;
  name: string;
  description: string | null;
  question_count: number;
  created_at: string;
  updated_at: string;
}

interface Round {
  id: number;
  topic: number;
  topic_name?: string;
  level: 'BEGINNER' | 'INTERMEDIATE' | 'ADVANCED';
  name: string;
  question_count: number;
  created_at: string;
  updated_at: string;
}

interface Question {
  id: number;
  topic: number;
  topic_name: string;
  round?: number | null;
  round_name?: string | null;
  source_type?: 'MANUAL' | 'LINK';
  source_type_display?: string;
  question_text: string;
  ideal_answer: string;
  difficulty: 'EASY' | 'MEDIUM' | 'HARD';
  is_active: boolean;
  reference_links?: string | null;
  reference_links_list?: string[];
  created_at: string;
  updated_at: string;
}

interface InterviewSession {
  id: number;
  user: number;
  user_email: string;
  user_name: string | null;
  started_at: string;
  ended_at: string | null;
  duration_seconds: number | null;
  topics: number[];
  topics_list: Array<{ id: number; name: string; question_count?: number }>;
  status: 'CREATED' | 'IN_PROGRESS' | 'COMPLETED' | 'CANCELLED';
  communication_score: number | null;
  technology_score: number | null;
  result_summary: string | null;
  answers: Answer[];
  answer_count: number;
  created_at: string;
  updated_at: string;
}

interface Answer {
  id: number;
  session: number;
  question: number;
  question_id: number;
  question_text: string;
  user_answer: string;
  similarity_score: number;
  accuracy_score?: number | null;
  completeness_score?: number | null;
  matched_keywords: string;
  missing_keywords: string;
  matched_keywords_list: string[];
  missing_keywords_list: string[];
  topic_score: number | null;
  communication_subscore: number | null;
  score_breakdown?: string | null;
  score_breakdown_dict?: {
    semantic_similarity?: number;
    keyword_coverage?: number;
    completeness?: number;
    communication_quality?: number;
    accuracy?: number;
  };
  created_at: string;
}

class ApiService {
  private async request<T>(
    endpoint: string,
    options: RequestInit = {}
  ): Promise<T> {
    const url = `${API_URL}${endpoint}`;
    const response = await fetch(url, {
      ...options,
      credentials: 'include', // Include cookies for session authentication
      headers: {
        'Content-Type': 'application/json',
        ...options.headers,
      },
    });

    if (!response.ok) {
      const errorData = await response.json().catch(() => ({ error: response.statusText }));
      // If there are detailed validation errors, include them
      // If there are detailed validation errors, include them
      // Handle various DRF error formats
      if (errorData.errors) {
        // Format: { errors: { field: ["error"] } }
        const errorMessages = Object.entries(errorData.errors)
          .map(([field, messages]) => `${field}: ${Array.isArray(messages) ? messages.join(', ') : messages} `)
          .join('; ');
        throw new Error(errorMessages || errorData.detail || errorData.error || `HTTP error! status: ${response.status} `);
      } else if (typeof errorData === 'object') {
        // Format: { field: ["error"] } or { error: "message" }
        // Filter out non-error common fields if any
        const messages = [];
        for (const [key, value] of Object.entries(errorData)) {
          if (key === 'error' || key === 'detail') {
            messages.push(String(value));
          } else if (Array.isArray(value)) {
            messages.push(`${key}: ${value.join(', ')}`);
          } else if (typeof value === 'string') {
            messages.push(`${key}: ${value}`);
          }
        }
        if (messages.length > 0) {
          throw new Error(messages.join('; '));
        }
      }

      throw new Error(errorData.error || errorData.detail || `HTTP error! status: ${response.status} `);
      throw new Error(errorData.error || errorData.detail || `HTTP error! status: ${response.status} `);
    }

    return response.json();
  }

  // User endpoints
  async login(username: string, password: string): Promise<UserProfile> {
    return this.request<UserProfile>('/users/login/', {
      method: 'POST',
      body: JSON.stringify({
        username,
        password,
      }),
    });
  }

  async createUser(userData: {
    username: string;
    password: string;
    email: string;
    name?: string;
    is_active?: boolean;
    role?: 'ADMIN' | 'USER';
    access_type?: 'TRIAL' | 'FULL' | null;
  }): Promise<UserProfile> {
    return this.request<UserProfile>('/users/', {
      method: 'POST',
      body: JSON.stringify(userData),
    });
  }

  async getUsers(): Promise<UserProfile[]> {
    try {
      const result = await this.request<any>('/users/');
      // Handle both direct array and paginated response
      if (Array.isArray(result)) {
        return result;
      } else if (result && Array.isArray(result.results)) {
        // Handle paginated response
        return result.results;
      } else {
        console.error('Invalid users response format:', result);
        return [];
      }
    } catch (error) {
      console.error('Error fetching users:', error);
      return []; // Return empty array on error
    }
  }

  async checkTrial(username: string): Promise<{ has_used_trial: boolean; can_start_interview: boolean; access_type?: string }> {
    return this.request(`/users/${username}/check-trial/`);
  }

  // Course endpoints
  async getCourses(): Promise<Course[]> {
    try {
      const result = await this.request<any>('/topics/');
      // Handle both direct array and paginated response
      if (Array.isArray(result)) {
        return result;
      } else if (result && Array.isArray(result.results)) {
        // Handle paginated response
        return result.results;
      } else {
        console.error('Invalid courses response format:', result);
        return [];
      }
    } catch (error) {
      console.error('Error fetching courses:', error);
      return []; // Return empty array on error
    }
  }

  // Question endpoints
  async getQuestions(topicId?: number, difficulty?: string): Promise<Question[]> {
    try {
      const params = new URLSearchParams();
      if (topicId) params.append('topic_id', topicId.toString());
      if (difficulty) params.append('difficulty', difficulty);

      const query = params.toString();
      const result = await this.request<any>(`/questions/${query ? `?${query}` : ''}`);

      // Handle both direct array and paginated response
      if (Array.isArray(result)) {
        return result;
      } else if (result && Array.isArray(result.results)) {
        // Handle paginated response
        return result.results;
      } else {
        console.error('Invalid questions response format:', result);
        return [];
      }
    } catch (error) {
      console.error('Error fetching questions:', error);
      return []; // Return empty array on error
    }
  }

  // Session endpoints
  async createSession(username: string, topicIds: number[]): Promise<InterviewSession> {
    return this.request<InterviewSession>('/sessions/', {
      method: 'POST',
      body: JSON.stringify({
        username,
        topic_ids: topicIds,
      }),
    });
  }

  async getSession(sessionId: number): Promise<InterviewSession> {
    return this.request<InterviewSession>(`/sessions/${sessionId}/`);
  }

  async getSessionsByUsername(username: string): Promise<InterviewSession[]> {
    try {
      const result = await this.request<any>(`/sessions/?username=${username}`);
      // Handle both direct array and paginated response
      if (Array.isArray(result)) {
        return result;
      } else if (result && Array.isArray(result.results)) {
        // Handle paginated response
        return result.results;
      } else {
        console.error('Invalid sessions response format:', result);
        return [];
      }
    } catch (error) {
      console.error('Error fetching sessions:', error);
      return []; // Return empty array on error
    }
  }

  async completeSession(sessionId: number): Promise<InterviewSession> {
    return this.request<InterviewSession>(`/sessions/${sessionId}/complete/`, {
      method: 'POST',
    });
  }

  async getSessionResults(sessionId: number): Promise<InterviewSession> {
    return this.request<InterviewSession>(`/sessions/${sessionId}/results/`);
  }

  // Answer endpoints
  async submitAnswer(sessionId: number, questionId: number, userAnswer: string): Promise<Answer> {
    return this.request<Answer>('/answers/', {
      method: 'POST',
      body: JSON.stringify({
        session: sessionId,
        question: questionId,
        user_answer: userAnswer,
      }),
    });
  }

  async getAnswers(sessionId: number): Promise<Answer[]> {
    return this.request<Answer[]>(`/answers/?session_id=${sessionId}`);
  }

  // Admin question endpoints
  async getAdminQuestions(topicId?: number, roundId?: number): Promise<Question[]> {
    try {
      const params = new URLSearchParams();
      if (topicId) params.append('topic_id', topicId.toString());
      if (roundId) params.append('round_id', roundId.toString());

      const query = params.toString();
      const result = await this.request<any>(`/admin/questions/${query ? `?${query}` : ''}`);

      if (Array.isArray(result)) {
        return result;
      } else if (result && Array.isArray(result.results)) {
        return result.results;
      }
      return [];
    } catch (error) {
      console.error('Error fetching admin questions:', error);
      return [];
    }
  }

  async createAdminQuestion(question: {
    topic: number;
    round?: number | null;
    source_type?: 'MANUAL' | 'LINK';
    question_text?: string;
    ideal_answer?: string;
    difficulty: 'EASY' | 'MEDIUM' | 'HARD';
    is_active?: boolean;
    reference_links?: string;
  }): Promise<Question> {
    return this.request<Question>('/admin/questions/', {
      method: 'POST',
      body: JSON.stringify(question),
    });
  }

  async updateAdminQuestion(id: number, question: Partial<Question>): Promise<Question> {
    return this.request<Question>(`/admin/questions/${id}/`, {
      method: 'PUT',
      body: JSON.stringify(question),
    });
  }

  async deleteAdminQuestion(id: number): Promise<void> {
    const url = `${API_URL}/admin/questions/${id}/`;
    const response = await fetch(url, {
      method: 'DELETE',
      credentials: 'include',
      headers: {
        'Content-Type': 'application/json',
      },
    });

    if (!response.ok) {
      const errorData = await response.json().catch(() => ({ error: response.statusText }));
      throw new Error(errorData.error || errorData.detail || `HTTP error! status: ${response.status}`);
    }

    // DELETE requests may return empty body (204 No Content)
    // Only try to parse JSON if there's content
    const contentType = response.headers.get('content-type');
    if (contentType && contentType.includes('application/json')) {
      const text = await response.text();
      if (text) {
        return JSON.parse(text);
      }
    }
    return;
  }

  // Admin course endpoints
  async getAdminCourses(): Promise<Course[]> {
    try {
      const result = await this.request<any>('/admin/topics/');
      if (Array.isArray(result)) {
        return result;
      } else if (result && Array.isArray(result.results)) {
        return result.results;
      }
      return [];
    } catch (error) {
      console.error('Error fetching admin courses:', error);
      return [];
    }
  }

  async createAdminCourse(course: {
    name: string;
    description?: string;
  }): Promise<Course> {
    return this.request<Course>('/admin/topics/', {
      method: 'POST',
      body: JSON.stringify(course),
    });
  }

  async updateAdminCourse(id: number, course: Partial<Course>): Promise<Course> {
    return this.request<Course>(`/admin/topics/${id}/`, {
      method: 'PUT',
      body: JSON.stringify(course),
    });
  }

  async deleteAdminCourse(id: number): Promise<void> {
    const url = `${API_URL}/admin/topics/${id}/`;
    const response = await fetch(url, {
      method: 'DELETE',
      credentials: 'include',
      headers: {
        'Content-Type': 'application/json',
      },
    });

    if (!response.ok) {
      const errorData = await response.json().catch(() => ({ error: response.statusText }));
      throw new Error(errorData.error || errorData.detail || `HTTP error! status: ${response.status}`);
    }

    // DELETE requests may return empty body (204 No Content)
    // Only try to parse JSON if there's content
    const contentType = response.headers.get('content-type');
    if (contentType && contentType.includes('application/json')) {
      const text = await response.text();
      if (text) {
        return JSON.parse(text);
      }
    }
    return;
  }

  // Admin Round endpoints
  async getAdminRounds(topicId?: number, level?: string): Promise<Round[]> {
    try {
      const params = new URLSearchParams();
      if (topicId) params.append('topic_id', topicId.toString());
      if (level) params.append('level', level);

      const query = params.toString();
      const result = await this.request<any>(`/admin/rounds/${query ? `?${query}` : ''}`);

      if (Array.isArray(result)) {
        return result;
      } else if (result && Array.isArray(result.results)) {
        return result.results;
      }
      return [];
    } catch (error) {
      console.error('Error fetching admin rounds:', error);
      return [];
    }
  }

  async createAdminRound(round: {
    topic: number;
    level: string;
    name: string;
  }): Promise<Round> {
    return this.request<Round>('/admin/rounds/', {
      method: 'POST',
      body: JSON.stringify(round),
    });
  }

  async updateAdminRound(id: number, round: Partial<Round>): Promise<Round> {
    return this.request<Round>(`/admin/rounds/${id}/`, {
      method: 'PUT',
      body: JSON.stringify(round),
    });
  }

  async deleteAdminRound(id: number): Promise<void> {
    const url = `${API_URL}/admin/rounds/${id}/`;
    const response = await fetch(url, {
      method: 'DELETE',
      credentials: 'include',
      headers: {
        'Content-Type': 'application/json',
      },
    });

    if (!response.ok) {
      const errorData = await response.json().catch(() => ({ error: response.statusText }));
      throw new Error(errorData.error || errorData.detail || `HTTP error! status: ${response.status}`);
    }

    // DELETE requests may return empty body (204 No Content)
    const contentType = response.headers.get('content-type');
    if (contentType && contentType.includes('application/json')) {
      const text = await response.text();
      if (text) {
        return JSON.parse(text);
      }
    }
    return;
  }

  // Admin user endpoints
  async updateAdminUser(id: number, userData: {
    is_active?: boolean;
    access_type?: 'TRIAL' | 'FULL';
    has_used_trial?: boolean;
    email?: string;
    name?: string;
    password?: string;
  }): Promise<UserProfile> {
    return this.request<UserProfile>(`/admin/users/${id}/`, {
      method: 'PATCH',
      body: JSON.stringify(userData),
    });
  }

  async deleteAdminUser(id: number): Promise<void> {
    const url = `${API_URL}/admin/users/${id}/`;
    const response = await fetch(url, {
      method: 'DELETE',
      credentials: 'include',
      headers: {
        'Content-Type': 'application/json',
      },
    });

    if (!response.ok) {
      const errorData = await response.json().catch(() => ({ error: response.statusText }));
      throw new Error(errorData.error || errorData.detail || `HTTP error! status: ${response.status}`);
    }

    // DELETE requests may return empty body (204 No Content)
    const contentType = response.headers.get('content-type');
    if (contentType && contentType.includes('application/json')) {
      const text = await response.text();
      if (text) {
        return JSON.parse(text);
      }
    }
    return;
  }

  async resetUserTrial(id: number): Promise<UserProfile> {
    return this.request<UserProfile>(`/admin/users/${id}/reset-trial/`, {
      method: 'POST',
    });
  }

  async toggleUserStatus(id: number): Promise<UserProfile> {
    return this.request<UserProfile>(`/admin/users/${id}/toggle-status/`, {
      method: 'POST',
    });
  }
}

export const apiService = new ApiService();
export type { UserProfile, Course, Round, Question, InterviewSession, Answer };

