import { useState, useEffect } from "react";
import { Button } from "@/components/ui/button";
import { Card } from "@/components/ui/card";
import { Input } from "@/components/ui/input";
import { Textarea } from "@/components/ui/textarea";
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogHeader,
  DialogTitle,
} from "@/components/ui/dialog";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import {
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
} from "@/components/ui/table";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { Plus, Edit, Trash2, Users, FileText, Eye, EyeOff, RefreshCw, ExternalLink, BookOpen, Lock } from "lucide-react";
import { apiService, type Question, type Topic } from "@/lib/api";
import { useToast } from "@/hooks/use-toast";

interface UserProfile {
  id: number;
  username: string;
  email: string;
  name: string | null;
  is_active: boolean;
  role: 'ADMIN' | 'USER';
  access_type: 'TRIAL' | 'FULL' | null;
  plain_password?: string | null;
  has_used_trial: boolean;
  created_at: string;
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
  topics_list: Array<{ id: number; name: string }>;
  status: 'CREATED' | 'IN_PROGRESS' | 'COMPLETED' | 'CANCELLED';
  communication_score: number | null;
  technology_score: number | null;
  result_summary: string | null;
  answer_count: number;
  created_at: string;
  updated_at: string;
}

const AdminDashboard = () => {
  const { toast } = useToast();
  const [showQuestionForm, setShowQuestionForm] = useState(false);
  const [showUserForm, setShowUserForm] = useState(false);
  const [users, setUsers] = useState<UserProfile[]>([]);
  const [isLoadingUsers, setIsLoadingUsers] = useState(false);

  // User form state
  const [userFormData, setUserFormData] = useState({
    username: "",
    password: "",
    email: "",
    name: "",
    is_active: true,
    role: 'USER' as 'ADMIN' | 'USER',
    access_type: 'TRIAL' as 'TRIAL' | 'FULL',
  });
  const [showPassword, setShowPassword] = useState(false);
  const [isCreatingUser, setIsCreatingUser] = useState(false);

  // User details dialog state
  const [selectedUser, setSelectedUser] = useState<UserProfile | null>(null);
  const [showUserDetails, setShowUserDetails] = useState(false);
  const [userSessions, setUserSessions] = useState<InterviewSession[]>([]);
  const [isLoadingSessions, setIsLoadingSessions] = useState(false);
  const [newPassword, setNewPassword] = useState("");
  const [showPasswordChange, setShowPasswordChange] = useState(false);

  // Questions state
  const [questions, setQuestions] = useState<Question[]>([]);
  const [topics, setTopics] = useState<Topic[]>([]);
  const [isLoadingQuestions, setIsLoadingQuestions] = useState(false);
  const [questionFormData, setQuestionFormData] = useState({
    source_type: "MANUAL" as "MANUAL" | "LINK",
    topic: "",
    question_text: "",
    ideal_answer: "",
    difficulty: "MEDIUM" as "EASY" | "MEDIUM" | "HARD",
    reference_links: "",
    is_active: true,
  });
  const [editingQuestion, setEditingQuestion] = useState<Question | null>(null);

  // Topics state
  const [isLoadingTopics, setIsLoadingTopics] = useState(false);
  const [showTopicForm, setShowTopicForm] = useState(false);
  const [topicFormData, setTopicFormData] = useState({
    name: "",
    description: "",
  });
  const [editingTopic, setEditingTopic] = useState<Topic | null>(null);

  // Fetch users on component mount
  useEffect(() => {
    fetchUsers();
    fetchQuestions();
    fetchTopics();
  }, []);

  // Fetch user sessions when viewing details
  useEffect(() => {
    if (selectedUser && showUserDetails) {
      fetchUserSessions(selectedUser.username);
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [selectedUser, showUserDetails]);

  const fetchUsers = async () => {
    setIsLoadingUsers(true);
    try {
      const fetchedUsers = await apiService.getUsers();
      // Ensure we always have an array
      if (Array.isArray(fetchedUsers)) {
        setUsers(fetchedUsers);
      } else {
        console.error('Invalid users response format:', fetchedUsers);
        setUsers([]);
        toast({
          title: "Error",
          description: "Invalid response format from server.",
          variant: "destructive",
        });
      }
    } catch (error: any) {
      console.error('Error fetching users:', error);
      setUsers([]); // Ensure users is always an array
      toast({
        title: "Error",
        description: error.message || "Failed to load users. Please try again.",
        variant: "destructive",
      });
    } finally {
      setIsLoadingUsers(false);
    }
  };

  const handleViewUserDetails = (user: UserProfile) => {
    setSelectedUser(user);
    setShowUserDetails(true);
  };

  const fetchUserSessions = async (username: string) => {
    setIsLoadingSessions(true);
    try {
      const sessions = await apiService.getSessionsByUsername(username);
      // Ensure we always have an array
      if (Array.isArray(sessions)) {
        setUserSessions(sessions);
      } else {
        console.error('Invalid sessions response format:', sessions);
        setUserSessions([]);
      }
    } catch (error: any) {
      console.error('Error fetching user sessions:', error);
      setUserSessions([]);
      toast({
        title: "Error",
        description: error.message || "Failed to load user sessions.",
        variant: "destructive",
      });
    } finally {
      setIsLoadingSessions(false);
    }
  };

  const handleResetTrial = async (userId: number) => {
    if (!confirm("Are you sure you want to reset the trial status for this user?")) return;

    try {
      await apiService.resetUserTrial(userId);
      toast({
        title: "Success",
        description: "Trial status has been reset successfully",
      });
      fetchUsers();
      if (selectedUser && selectedUser.id === userId) {
        // Refresh selected user data
        const updatedUsers = await apiService.getUsers();
        const updatedUser = updatedUsers.find(u => u.id === userId);
        if (updatedUser) {
          setSelectedUser(updatedUser);
        }
      }
    } catch (error: any) {
      toast({
        title: "Error",
        description: error.message || "Failed to reset trial status",
        variant: "destructive",
      });
    }
  };

  const handleToggleUserStatus = async (user: UserProfile) => {
    try {
      await apiService.toggleUserStatus(user.id);
      toast({
        title: "Success",
        description: `User ${user.is_active ? 'deactivated' : 'activated'} successfully`,
      });
      fetchUsers();
      if (selectedUser && selectedUser.id === user.id) {
        // Refresh selected user data
        const updatedUsers = await apiService.getUsers();
        const updatedUser = updatedUsers.find(u => u.id === user.id);
        if (updatedUser) {
          setSelectedUser(updatedUser);
        }
      }
    } catch (error: any) {
      toast({
        title: "Error",
        description: error.message || "Failed to update user status",
        variant: "destructive",
      });
    }
  };

  const handleDeleteUser = async (userId: number) => {
    if (!confirm("Are you sure you want to delete this user? This action cannot be undone.")) return;

    try {
      await apiService.deleteAdminUser(userId);
      toast({
        title: "Success",
        description: "User deleted successfully",
      });
      fetchUsers();
      if (selectedUser && selectedUser.id === userId) {
        setShowUserDetails(false);
        setSelectedUser(null);
      }
    } catch (error: any) {
      toast({
        title: "Error",
        description: error.message || "Failed to delete user",
        variant: "destructive",
      });
    }
  };

  const handleUpdateAccessType = async () => {
    if (!selectedUser) return;

    const newAccessType = selectedUser.access_type === 'FULL' ? 'TRIAL' : 'FULL';

    try {
      await apiService.updateAdminUser(selectedUser.id, {
        access_type: newAccessType
      });

      setSelectedUser({ ...selectedUser, access_type: newAccessType });
      toast({
        title: "Success",
        description: `User access updated to ${newAccessType === 'FULL' ? 'Full Access' : 'Trial'}`,
      });
      fetchUsers();
    } catch (error: any) {
      toast({
        title: "Error",
        description: error.message || "Failed to update access type",
        variant: "destructive",
      });
    }
  };

  const handleChangePassword = async () => {
    if (!selectedUser || !newPassword) return;

    try {
      await apiService.updateAdminUser(selectedUser.id, {
        password: newPassword
      });

      toast({
        title: "Success",
        description: "Password updated successfully",
      });
      setNewPassword("");
      setShowPasswordChange(false);
    } catch (error: any) {
      toast({
        title: "Error",
        description: error.message || "Failed to update password",
        variant: "destructive",
      });
    }
  };

  const handleCreateUser = async () => {
    if (!userFormData.username || !userFormData.password || !userFormData.email) {
      toast({
        title: "Missing Fields",
        description: "Please fill in username, password, and email.",
        variant: "destructive",
      });
      return;
    }

    setIsCreatingUser(true);
    try {
      await apiService.createUser({
        username: userFormData.username,
        password: userFormData.password,
        email: userFormData.email,
        name: userFormData.name || undefined,
        is_active: userFormData.is_active,
        role: userFormData.role,
        access_type: userFormData.role === 'USER' ? userFormData.access_type : undefined,
      });

      toast({
        title: "User Created",
        description: `User "${userFormData.username}" has been created successfully.`,
        variant: "success",
      });

      // Reset form
      setUserFormData({
        username: "",
        password: "",
        email: "",
        name: "",
        is_active: true,
        role: 'USER',
        access_type: 'TRIAL',
      });
      setShowUserForm(false);

      // Refresh users list
      fetchUsers();
    } catch (error: any) {
      console.error('Error creating user:', error);
      toast({
        title: "Error",
        description: error.message || "Failed to create user. Please try again.",
        variant: "destructive",
      });
    } finally {
      setIsCreatingUser(false);
    }
  };

  const fetchQuestions = async () => {
    setIsLoadingQuestions(true);
    try {
      const data = await apiService.getAdminQuestions();
      setQuestions(data);
    } catch (error) {
      console.error("Error fetching questions:", error);
      toast({
        title: "Error",
        description: (error as Error).message || "Failed to fetch questions",
        variant: "destructive",
      });
    } finally {
      setIsLoadingQuestions(false);
    }
  };

  const fetchTopics = async () => {
    setIsLoadingTopics(true);
    try {
      const data = await apiService.getAdminTopics();
      setTopics(data);
    } catch (error) {
      console.error("Error fetching topics:", error);
      toast({
        title: "Error",
        description: (error as Error).message || "Failed to fetch topics",
        variant: "destructive",
      });
    } finally {
      setIsLoadingTopics(false);
    }
  };

  const handleCreateTopic = async () => {
    if (!topicFormData.name || !topicFormData.name.trim()) {
      toast({
        title: "Validation Error",
        description: "Topic name is required",
        variant: "destructive",
      });
      return;
    }

    try {
      if (editingTopic) {
        await apiService.updateAdminTopic(editingTopic.id, topicFormData);
        toast({
          title: "Success",
          description: "Topic updated successfully",
        });
      } else {
        await apiService.createAdminTopic(topicFormData);
        toast({
          title: "Success",
          description: "Topic created successfully",
        });
      }
      setShowTopicForm(false);
      setEditingTopic(null);
      setTopicFormData({ name: "", description: "" });
      fetchTopics();
      // Also refresh topics for question form
      const allTopics = await apiService.getTopics();
      setTopics(allTopics);
    } catch (error: any) {
      toast({
        title: "Error",
        description: error.message || "Failed to save topic",
        variant: "destructive",
      });
    }
  };

  const handleDeleteTopic = async (id: number) => {
    if (!confirm("Are you sure you want to delete this topic? This will also delete all questions associated with it.")) return;

    try {
      await apiService.deleteAdminTopic(id);
      toast({
        title: "Success",
        description: "Topic deleted successfully",
      });
      fetchTopics();
      // Refresh topics for question form
      const allTopics = await apiService.getTopics();
      setTopics(allTopics);
    } catch (error: any) {
      toast({
        title: "Error",
        description: error.message || "Failed to delete topic",
        variant: "destructive",
      });
    }
  };

  const handleCreateQuestion = async () => {
    // Validate based on source_type
    if (!questionFormData.topic) {
      toast({
        title: "Validation Error",
        description: "Please select a topic",
        variant: "destructive",
      });
      return;
    }

    if (questionFormData.source_type === "MANUAL") {
      if (!questionFormData.question_text || !questionFormData.ideal_answer) {
        toast({
          title: "Validation Error",
          description: "Question text and ideal answer are required for manual definition",
          variant: "destructive",
        });
        return;
      }
    } else if (questionFormData.source_type === "LINK") {
      if (!questionFormData.reference_links || !questionFormData.reference_links.trim()) {
        toast({
          title: "Validation Error",
          description: "Reference links are required when using link-based definition",
          variant: "destructive",
        });
        return;
      }
    }

    try {
      await apiService.createAdminQuestion({
        topic: parseInt(questionFormData.topic),
        source_type: questionFormData.source_type,
        question_text: questionFormData.source_type === "MANUAL" ? questionFormData.question_text : undefined,
        ideal_answer: questionFormData.source_type === "MANUAL" ? questionFormData.ideal_answer : undefined,
        difficulty: questionFormData.difficulty,
        is_active: questionFormData.is_active,
        reference_links: questionFormData.source_type === "LINK" ? questionFormData.reference_links : (questionFormData.reference_links || undefined),
      });
      toast({
        title: "Success",
        description: "Question created successfully",
      });
      setShowQuestionForm(false);
      setEditingQuestion(null);
      setQuestionFormData({
        source_type: "MANUAL",
        topic: "",
        question_text: "",
        ideal_answer: "",
        difficulty: "MEDIUM",
        reference_links: "",
        is_active: true,
      });
      fetchQuestions();
    } catch (error: any) {
      toast({
        title: "Error",
        description: error.message || "Failed to create question",
        variant: "destructive",
      });
    }
  };

  const handleDeleteQuestion = async (id: number) => {
    if (!confirm("Are you sure you want to delete this question?")) return;

    try {
      await apiService.deleteAdminQuestion(id);
      toast({
        title: "Success",
        description: "Question deleted successfully",
      });
      fetchQuestions();
    } catch (error: any) {
      toast({
        title: "Error",
        description: error.message || "Failed to delete question",
        variant: "destructive",
      });
    }
  };

  return (
    <div className="min-h-screen bg-slate-950 text-slate-200 p-6">
      <div className="max-w-7xl mx-auto">
        <div className="mb-8">
          <h1 className="text-4xl font-bold text-white mb-2 tracking-tight">
            Admin Dashboard
          </h1>
          <p className="text-slate-400">
            Manage interviews, questions, and user results
          </p>
        </div>

        <Tabs defaultValue="users" className="space-y-6">
          <TabsList className="bg-slate-900 border border-slate-800">
            <TabsTrigger value="users" className="data-[state=active]:bg-slate-800 data-[state=active]:text-white text-slate-400">
              <Users className="h-4 w-4 mr-2" />
              Users & Results
            </TabsTrigger>
            <TabsTrigger value="topics" className="data-[state=active]:bg-slate-800 data-[state=active]:text-white text-slate-400">
              <BookOpen className="h-4 w-4 mr-2" />
              Topics
            </TabsTrigger>
            <TabsTrigger value="questions" className="data-[state=active]:bg-slate-800 data-[state=active]:text-white text-slate-400">
              <FileText className="h-4 w-4 mr-2" />
              Question Bank
            </TabsTrigger>
          </TabsList>

          <TabsContent value="users">
            <div className="space-y-4">
              <div className="flex justify-end">
                <Button
                  onClick={() => setShowUserForm(!showUserForm)}
                  className="bg-white text-slate-950 hover:bg-slate-200"
                >
                  <Plus className="h-4 w-4 mr-2" />
                  Add User
                </Button>
              </div>

              {showUserForm && (
                <Card className="p-6 bg-card/50 backdrop-blur-lg border-primary/20 animate-slide-up">
                  <h3 className="text-lg font-semibold text-foreground mb-4">
                    Create New User
                  </h3>
                  <div className="space-y-4">
                    <div>
                      <label className="text-sm text-foreground mb-2 block">
                        Username <span className="text-destructive">*</span>
                      </label>
                      <Input
                        placeholder="Enter username"
                        value={userFormData.username}
                        onChange={(e) => setUserFormData({ ...userFormData, username: e.target.value })}
                        className="bg-input border-border"
                      />
                    </div>
                    <div>
                      <label className="text-sm text-foreground mb-2 block">
                        Password <span className="text-destructive">*</span>
                      </label>
                      <div className="relative">
                        <Input
                          type={showPassword ? "text" : "password"}
                          placeholder="Enter password"
                          value={userFormData.password}
                          onChange={(e) => setUserFormData({ ...userFormData, password: e.target.value })}
                          className="bg-input border-border pr-10"
                        />
                        <button
                          type="button"
                          onClick={() => setShowPassword(!showPassword)}
                          className="absolute right-3 top-1/2 -translate-y-1/2 text-muted-foreground hover:text-foreground transition-colors focus:outline-none"
                        >
                          {showPassword ? (
                            <EyeOff className="h-4 w-4" />
                          ) : (
                            <Eye className="h-4 w-4" />
                          )}
                        </button>
                      </div>
                    </div>
                    <div>
                      <label className="text-sm text-foreground mb-2 block">
                        Email <span className="text-destructive">*</span>
                      </label>
                      <Input
                        type="email"
                        placeholder="Enter email"
                        value={userFormData.email}
                        onChange={(e) => setUserFormData({ ...userFormData, email: e.target.value })}
                        className="bg-input border-border"
                      />
                    </div>
                    <div>
                      <label className="text-sm text-foreground mb-2 block">
                        Name (Optional)
                      </label>
                      <Input
                        placeholder="Enter full name"
                        value={userFormData.name}
                        onChange={(e) => setUserFormData({ ...userFormData, name: e.target.value })}
                        className="bg-input border-border"
                      />
                    </div>
                    <div>
                      <label className="text-sm text-foreground mb-2 block">
                        Role <span className="text-destructive">*</span>
                      </label>
                      <Select
                        value={userFormData.role}
                        onValueChange={(value: 'ADMIN' | 'USER') =>
                          setUserFormData({ ...userFormData, role: value })
                        }
                      >
                        <SelectTrigger className="bg-input border-border text-foreground">
                          <SelectValue placeholder="Select role" />
                        </SelectTrigger>
                        <SelectContent>
                          <SelectItem value="USER">Student (User)</SelectItem>
                          <SelectItem value="ADMIN">Administrator</SelectItem>
                        </SelectContent>
                      </Select>
                    </div>

                    {userFormData.role === 'USER' && (
                      <div>
                        <label className="text-sm text-foreground mb-2 block">
                          Access Type <span className="text-destructive">*</span>
                        </label>
                        <Select
                          value={userFormData.access_type}
                          onValueChange={(value: 'TRIAL' | 'FULL') =>
                            setUserFormData({ ...userFormData, access_type: value })
                          }
                        >
                          <SelectTrigger className="bg-input border-border text-foreground">
                            <SelectValue placeholder="Select access type" />
                          </SelectTrigger>
                          <SelectContent>
                            <SelectItem value="TRIAL">Trial - One Free Interview</SelectItem>
                            <SelectItem value="FULL">Full Access</SelectItem>
                          </SelectContent>
                        </Select>
                        <p className="text-xs text-muted-foreground mt-1">
                          {userFormData.access_type === 'TRIAL'
                            ? 'User will get one free interview'
                            : 'User can create unlimited interview sessions'}
                        </p>
                      </div>
                    )}
                    <div className="flex items-center gap-2">
                      <input
                        type="checkbox"
                        id="is_active"
                        checked={userFormData.is_active}
                        onChange={(e) => setUserFormData({ ...userFormData, is_active: e.target.checked })}
                        className="w-4 h-4 rounded border-border bg-input text-primary focus:ring-primary"
                      />
                      <label htmlFor="is_active" className="text-sm text-foreground cursor-pointer">
                        Account is active
                      </label>
                    </div>
                    <div className="flex gap-2">
                      <Button
                        onClick={() => {
                          setShowUserForm(false);
                          setUserFormData({
                            username: "",
                            password: "",
                            email: "",
                            name: "",
                            is_active: true,
                            role: 'USER',
                            access_type: 'TRIAL',
                          });
                        }}
                        variant="outline"
                        className="flex-1"
                        disabled={isCreatingUser}
                      >
                        Cancel
                      </Button>
                      <Button
                        onClick={handleCreateUser}
                        className="flex-1 bg-primary hover:bg-primary/90 text-primary-foreground"
                        disabled={isCreatingUser}
                      >
                        {isCreatingUser ? "Creating..." : "Create User"}
                      </Button>
                    </div>
                  </div>
                </Card>
              )}

              <Card className="p-6 bg-card/30 backdrop-blur-lg border-border">
                {isLoadingUsers ? (
                  <div className="text-center py-8 text-muted-foreground">
                    Loading users...
                  </div>
                ) : users.length === 0 ? (
                  <div className="text-center py-8 text-muted-foreground">
                    No users found. Create your first user above.
                  </div>
                ) : (
                  <Table>
                    <TableHeader>
                      <TableRow className="border-border hover:bg-muted/20">
                        <TableHead className="text-foreground">Username</TableHead>
                        <TableHead className="text-foreground">Name</TableHead>
                        <TableHead className="text-foreground">Email</TableHead>
                        <TableHead className="text-foreground">Role</TableHead>
                        <TableHead className="text-foreground">Access Type</TableHead>
                        <TableHead className="text-foreground">Status</TableHead>
                        <TableHead className="text-foreground">Trial Used</TableHead>
                        <TableHead className="text-foreground">Created</TableHead>
                        <TableHead className="text-foreground">Actions</TableHead>
                      </TableRow>
                    </TableHeader>
                    <TableBody>
                      {Array.isArray(users) && users.map((user) => (
                        <TableRow key={user.id} className="border-border hover:bg-muted/20">
                          <TableCell className="text-foreground font-medium">{user.username}</TableCell>
                          <TableCell className="text-foreground">{user.name || "-"}</TableCell>
                          <TableCell className="text-muted-foreground">{user.email}</TableCell>
                          <TableCell>
                            <span className={`px-2 py-1 rounded text-xs ${user.role === 'ADMIN' ? 'bg-purple-500/20 text-purple-500' : 'bg-blue-500/20 text-blue-500'}`}>
                              {user.role === 'ADMIN' ? 'Admin' : 'User'}
                            </span>
                          </TableCell>
                          <TableCell>
                            <span
                              className={`px-2 py-1 rounded text-xs ${user.access_type === 'FULL'
                                ? "bg-primary/20 text-primary font-medium"
                                : "bg-secondary/20 text-secondary"
                                }`}
                            >
                              {user.access_type === 'FULL' ? 'Full Access' : 'Trial'}
                            </span>
                          </TableCell>
                          <TableCell>
                            <span
                              className={`px-2 py-1 rounded text-xs ${user.is_active
                                ? "bg-primary/20 text-primary"
                                : "bg-muted text-muted-foreground"
                                }`}
                            >
                              {user.is_active ? "Active" : "Inactive"}
                            </span>
                          </TableCell>
                          <TableCell>
                            <span
                              className={`px-2 py-1 rounded text-xs ${user.has_used_trial
                                ? "bg-secondary/20 text-secondary"
                                : "bg-primary/20 text-primary"
                                }`}
                            >
                              {user.has_used_trial ? "Yes" : "No"}
                            </span>
                          </TableCell>
                          <TableCell className="text-muted-foreground text-sm">
                            {new Date(user.created_at).toLocaleDateString()}
                          </TableCell>
                          <TableCell>
                            <div className="flex gap-2">
                              <Button
                                variant="ghost"
                                size="sm"
                                className="hover:bg-primary/10"
                                onClick={() => handleViewUserDetails(user)}
                              >
                                View Details
                              </Button>
                              <Button
                                variant="ghost"
                                size="sm"
                                className="hover:bg-destructive/10"
                                onClick={() => handleDeleteUser(user.id)}
                              >
                                <Trash2 className="h-4 w-4 text-destructive" />
                              </Button>
                            </div>
                          </TableCell>
                        </TableRow>
                      ))}
                    </TableBody>
                  </Table>
                )}
              </Card>
            </div>
          </TabsContent>

          <TabsContent value="topics">
            <div className="space-y-4">
              <div className="flex justify-end">
                <Button
                  onClick={() => setShowTopicForm(!showTopicForm)}
                  className="bg-primary hover:bg-primary/90 text-primary-foreground glow-cyan"
                >
                  <Plus className="h-4 w-4 mr-2" />
                  Add Topic
                </Button>
              </div>

              {showTopicForm && (
                <Card className="p-6 bg-card/50 backdrop-blur-lg border-primary/20 animate-slide-up">
                  <h3 className="text-lg font-semibold text-foreground mb-4">
                    {editingTopic ? "Edit Topic" : "Add New Topic"}
                  </h3>
                  <div className="space-y-4">
                    <div>
                      <label className="text-sm text-foreground mb-2 block">Topic Name *</label>
                      <Input
                        placeholder="e.g., Machine Learning, React, System Design"
                        className="bg-input border-border text-foreground"
                        value={topicFormData.name}
                        onChange={(e) => setTopicFormData({ ...topicFormData, name: e.target.value })}
                      />
                    </div>
                    <div>
                      <label className="text-sm text-foreground mb-2 block">Description (Optional)</label>
                      <Textarea
                        placeholder="Brief description of this topic..."
                        className="bg-input border-border min-h-[80px]"
                        value={topicFormData.description}
                        onChange={(e) => setTopicFormData({ ...topicFormData, description: e.target.value })}
                      />
                    </div>
                    <div className="flex gap-2">
                      <Button
                        onClick={() => {
                          setShowTopicForm(false);
                          setEditingTopic(null);
                          setTopicFormData({ name: "", description: "" });
                        }}
                        variant="outline"
                        className="flex-1"
                      >
                        Cancel
                      </Button>
                      <Button
                        onClick={handleCreateTopic}
                        className="flex-1 bg-primary hover:bg-primary/90 text-primary-foreground"
                      >
                        {editingTopic ? "Update Topic" : "Save Topic"}
                      </Button>
                    </div>
                  </div>
                </Card>
              )}

              <Card className="p-6 bg-card/30 backdrop-blur-lg border-border">
                {isLoadingTopics ? (
                  <div className="text-center py-8 text-muted-foreground">Loading topics...</div>
                ) : topics.length === 0 ? (
                  <div className="text-center py-8 text-muted-foreground">No topics found. Add your first topic above.</div>
                ) : (
                  <Table>
                    <TableHeader>
                      <TableRow className="border-border hover:bg-muted/20">
                        <TableHead className="text-foreground">Topic Name</TableHead>
                        <TableHead className="text-foreground">Description</TableHead>
                        <TableHead className="text-foreground">Questions</TableHead>
                        <TableHead className="text-foreground">Created</TableHead>
                        <TableHead className="text-foreground">Actions</TableHead>
                      </TableRow>
                    </TableHeader>
                    <TableBody>
                      {topics.map((topic) => (
                        <TableRow key={topic.id} className="border-border hover:bg-muted/20">
                          <TableCell className="text-foreground font-medium">{topic.name}</TableCell>
                          <TableCell className="text-foreground">
                            {topic.description || <span className="text-muted-foreground">No description</span>}
                          </TableCell>
                          <TableCell>
                            <span className="px-2 py-1 rounded text-xs bg-primary/20 text-primary">
                              {topic.question_count || 0} questions
                            </span>
                          </TableCell>
                          <TableCell className="text-muted-foreground text-sm">
                            {new Date(topic.created_at).toLocaleDateString()}
                          </TableCell>
                          <TableCell>
                            <div className="flex gap-2">
                              <Button
                                variant="ghost"
                                size="sm"
                                className="hover:bg-primary/10"
                                onClick={() => {
                                  setEditingTopic(topic);
                                  setTopicFormData({
                                    name: topic.name,
                                    description: topic.description || "",
                                  });
                                  setShowTopicForm(true);
                                }}
                              >
                                <Edit className="h-4 w-4" />
                              </Button>
                              <Button
                                variant="ghost"
                                size="sm"
                                className="hover:bg-destructive/10"
                                onClick={() => handleDeleteTopic(topic.id)}
                              >
                                <Trash2 className="h-4 w-4 text-destructive" />
                              </Button>
                            </div>
                          </TableCell>
                        </TableRow>
                      ))}
                    </TableBody>
                  </Table>
                )}
              </Card>
            </div>
          </TabsContent>

          <TabsContent value="questions">
            <div className="space-y-4">
              <div className="flex justify-end">
                <Button
                  onClick={() => setShowQuestionForm(!showQuestionForm)}
                  className="bg-primary hover:bg-primary/90 text-primary-foreground glow-cyan"
                >
                  <Plus className="h-4 w-4 mr-2" />
                  Add Question
                </Button>
              </div>

              {showQuestionForm && (
                <Card className="p-6 bg-card/50 backdrop-blur-lg border-primary/20 animate-slide-up">
                  <h3 className="text-lg font-semibold text-foreground mb-4">
                    {editingQuestion ? "Edit Question" : "Add New Question"}
                  </h3>
                  <div className="space-y-4">
                    <div>
                      <label className="text-sm text-foreground mb-2 block">Source Type *</label>
                      <Select
                        value={questionFormData.source_type}
                        onValueChange={(value: "MANUAL" | "LINK") =>
                          setQuestionFormData({ ...questionFormData, source_type: value })
                        }
                      >
                        <SelectTrigger className="bg-input border-border text-foreground">
                          <SelectValue />
                        </SelectTrigger>
                        <SelectContent>
                          <SelectItem value="MANUAL">Manually Define Q&A</SelectItem>
                          <SelectItem value="LINK">Use External Links</SelectItem>
                        </SelectContent>
                      </Select>
                      <p className="text-xs text-muted-foreground mt-1">
                        {questionFormData.source_type === "MANUAL"
                          ? "Enter the question and answer directly"
                          : "Provide links to websites containing questions and answers"}
                      </p>
                    </div>
                    <div>
                      <label className="text-sm text-foreground mb-2 block">Topic *</label>
                      <Select
                        value={questionFormData.topic}
                        onValueChange={(value) => setQuestionFormData({ ...questionFormData, topic: value })}
                      >
                        <SelectTrigger className="bg-input border-border text-foreground">
                          <SelectValue placeholder="Select a topic" />
                        </SelectTrigger>
                        <SelectContent>
                          {topics.map((topic) => (
                            <SelectItem key={topic.id} value={topic.id.toString()}>
                              {topic.name}
                            </SelectItem>
                          ))}
                        </SelectContent>
                      </Select>
                    </div>
                    {questionFormData.source_type === "MANUAL" && (
                      <>
                        <div>
                          <label className="text-sm text-foreground mb-2 block">Question *</label>
                          <Textarea
                            placeholder="Enter the interview question..."
                            className="bg-input border-border min-h-[100px]"
                            value={questionFormData.question_text}
                            onChange={(e) => setQuestionFormData({ ...questionFormData, question_text: e.target.value })}
                          />
                        </div>
                        <div>
                          <label className="text-sm text-foreground mb-2 block">Ideal Answer *</label>
                          <Textarea
                            placeholder="Enter the ideal answer with key points..."
                            className="bg-input border-border min-h-[100px]"
                            value={questionFormData.ideal_answer}
                            onChange={(e) => setQuestionFormData({ ...questionFormData, ideal_answer: e.target.value })}
                          />
                        </div>
                      </>
                    )}
                    <div>
                      <label className="text-sm text-foreground mb-2 block">Difficulty *</label>
                      <Select
                        value={questionFormData.difficulty}
                        onValueChange={(value: "EASY" | "MEDIUM" | "HARD") =>
                          setQuestionFormData({ ...questionFormData, difficulty: value })
                        }
                      >
                        <SelectTrigger className="bg-input border-border text-foreground">
                          <SelectValue />
                        </SelectTrigger>
                        <SelectContent>
                          <SelectItem value="EASY">Easy</SelectItem>
                          <SelectItem value="MEDIUM">Medium</SelectItem>
                          <SelectItem value="HARD">Hard</SelectItem>
                        </SelectContent>
                      </Select>
                    </div>
                    {questionFormData.source_type === "LINK" && (
                      <div>
                        <label className="text-sm text-foreground mb-2 block">
                          Reference Links *
                        </label>
                        <Textarea
                          placeholder="Enter one URL per line. These links contain the questions and answers for the interview.&#10;Example:&#10;https://example.com/python-interview-questions&#10;https://example.com/python-answers"
                          className="bg-input border-border min-h-[100px] font-mono text-sm"
                          value={questionFormData.reference_links}
                          onChange={(e) => setQuestionFormData({ ...questionFormData, reference_links: e.target.value })}
                        />
                        <p className="text-xs text-muted-foreground mt-1">
                          Required when using link-based definition. Add URLs to websites containing relevant questions and answers. One URL per line.
                        </p>
                      </div>
                    )}
                    {questionFormData.source_type === "MANUAL" && (
                      <div>
                        <label className="text-sm text-foreground mb-2 block">
                          Reference Links (Optional)
                        </label>
                        <Textarea
                          placeholder="Optional: Add URLs for additional reference material.&#10;Example:&#10;https://example.com/python-docs"
                          className="bg-input border-border min-h-[80px] font-mono text-sm"
                          value={questionFormData.reference_links}
                          onChange={(e) => setQuestionFormData({ ...questionFormData, reference_links: e.target.value })}
                        />
                        <p className="text-xs text-muted-foreground mt-1">
                          Optional: Add URLs to websites with additional reference material. One URL per line.
                        </p>
                      </div>
                    )}
                    <div className="flex gap-2">
                      <Button
                        onClick={() => {
                          setShowQuestionForm(false);
                          setEditingQuestion(null);
                          setQuestionFormData({
                            source_type: "MANUAL",
                            topic: "",
                            question_text: "",
                            ideal_answer: "",
                            difficulty: "MEDIUM",
                            reference_links: "",
                            is_active: true,
                          });
                        }}
                        variant="outline"
                        className="flex-1"
                      >
                        Cancel
                      </Button>
                      <Button
                        onClick={handleCreateQuestion}
                        className="flex-1 bg-primary hover:bg-primary/90 text-primary-foreground"
                      >
                        {editingQuestion ? "Update Question" : "Save Question"}
                      </Button>
                    </div>
                  </div>
                </Card>
              )}

              <Card className="p-6 bg-card/30 backdrop-blur-lg border-border">
                {isLoadingQuestions ? (
                  <div className="text-center py-8 text-muted-foreground">Loading questions...</div>
                ) : questions.length === 0 ? (
                  <div className="text-center py-8 text-muted-foreground">No questions found. Add your first question above.</div>
                ) : (
                  <Table>
                    <TableHeader>
                      <TableRow className="border-border hover:bg-muted/20">
                        <TableHead className="text-foreground">Topic</TableHead>
                        <TableHead className="text-foreground">Source</TableHead>
                        <TableHead className="text-foreground">Question</TableHead>
                        <TableHead className="text-foreground">Difficulty</TableHead>
                        <TableHead className="text-foreground">Reference Links</TableHead>
                        <TableHead className="text-foreground">Status</TableHead>
                        <TableHead className="text-foreground">Actions</TableHead>
                      </TableRow>
                    </TableHeader>
                    <TableBody>
                      {questions.map((q) => (
                        <TableRow key={q.id} className="border-border hover:bg-muted/20">
                          <TableCell className="text-foreground">{q.topic_name}</TableCell>
                          <TableCell>
                            <span
                              className={`px-2 py-1 rounded text-xs ${q.source_type === "LINK"
                                ? "bg-blue-500/20 text-blue-500"
                                : "bg-purple-500/20 text-purple-500"
                                }`}
                            >
                              {q.source_type === "LINK" ? "From Links" : "Manual"}
                            </span>
                          </TableCell>
                          <TableCell className="text-foreground max-w-md">
                            <div className="truncate" title={q.question_text || "No preview text"}>
                              {q.question_text || (q.source_type === "LINK" ? "[From Links]" : "[No text]")}
                            </div>
                          </TableCell>
                          <TableCell>
                            <span
                              className={`px-2 py-1 rounded text-xs ${q.difficulty === "HARD"
                                ? "bg-destructive/20 text-destructive"
                                : q.difficulty === "MEDIUM"
                                  ? "bg-secondary/20 text-secondary"
                                  : "bg-primary/20 text-primary"
                                }`}
                            >
                              {q.difficulty}
                            </span>
                          </TableCell>
                          <TableCell>
                            {q.reference_links_list && q.reference_links_list.length > 0 ? (
                              <div className="flex flex-col gap-1">
                                {q.reference_links_list.slice(0, 2).map((link, idx) => (
                                  <a
                                    key={idx}
                                    href={link}
                                    target="_blank"
                                    rel="noopener noreferrer"
                                    className="text-xs text-primary hover:underline flex items-center gap-1"
                                  >
                                    <ExternalLink className="h-3 w-3" />
                                    {link.length > 40 ? `${link.substring(0, 40)}...` : link}
                                  </a>
                                ))}
                                {q.reference_links_list.length > 2 && (
                                  <span className="text-xs text-muted-foreground">
                                    +{q.reference_links_list.length - 2} more
                                  </span>
                                )}
                              </div>
                            ) : (
                              <span className="text-xs text-muted-foreground">No links</span>
                            )}
                          </TableCell>
                          <TableCell>
                            <span
                              className={`px-2 py-1 rounded text-xs ${q.is_active
                                ? "bg-green-500/20 text-green-500"
                                : "bg-gray-500/20 text-gray-500"
                                }`}
                            >
                              {q.is_active ? "Active" : "Inactive"}
                            </span>
                          </TableCell>
                          <TableCell>
                            <div className="flex gap-2">
                              <Button
                                variant="ghost"
                                size="sm"
                                className="hover:bg-primary/10"
                                onClick={() => {
                                  setEditingQuestion(q);
                                  setQuestionFormData({
                                    source_type: q.source_type || "MANUAL",
                                    topic: q.topic.toString(),
                                    question_text: q.question_text || "",
                                    ideal_answer: q.ideal_answer || "",
                                    difficulty: q.difficulty,
                                    reference_links: q.reference_links || "",
                                    is_active: q.is_active,
                                  });
                                  setShowQuestionForm(true);
                                }}
                              >
                                <Edit className="h-4 w-4" />
                              </Button>
                              <Button
                                variant="ghost"
                                size="sm"
                                className="hover:bg-destructive/10"
                                onClick={() => handleDeleteQuestion(q.id)}
                              >
                                <Trash2 className="h-4 w-4 text-destructive" />
                              </Button>
                            </div>
                          </TableCell>
                        </TableRow>
                      ))}
                    </TableBody>
                  </Table>
                )}
              </Card>
            </div>
          </TabsContent>
        </Tabs>
      </div>

      {/* User Details Dialog */}
      <Dialog open={showUserDetails} onOpenChange={setShowUserDetails}>
        <DialogContent className="max-w-4xl max-h-[90vh] overflow-y-auto bg-card/95 backdrop-blur-lg border-primary/20">
          <DialogHeader>
            <DialogTitle className="text-2xl font-bold text-gradient-primary">
              User Details
            </DialogTitle>
            <DialogDescription>
              View and manage user information and interview sessions
            </DialogDescription>
          </DialogHeader>

          {selectedUser && (
            <div className="space-y-6 mt-4">
              {/* User Information */}
              <Card className="p-4 bg-card/50 border-border">
                <h3 className="text-lg font-semibold text-foreground mb-4">User Information</h3>
                <div className="grid grid-cols-2 gap-4">
                  <div>
                    <label className="text-sm text-muted-foreground">Username</label>
                    <p className="text-foreground font-medium">{selectedUser.username}</p>
                  </div>
                  <div>
                    <label className="text-sm text-muted-foreground">Email</label>
                    <p className="text-foreground">{selectedUser.email}</p>
                  </div>
                  <div>
                    <label className="text-sm text-muted-foreground">Name</label>
                    <p className="text-foreground">{selectedUser.name || "-"}</p>
                  </div>
                  <div>
                    <label className="text-sm text-muted-foreground">Account Status</label>
                    <p className="text-foreground">
                      <span
                        className={`px-2 py-1 rounded text-xs ${selectedUser.is_active
                          ? "bg-primary/20 text-primary"
                          : "bg-muted text-muted-foreground"
                          }`}
                      >
                        {selectedUser.is_active ? "Active" : "Inactive"}
                      </span>
                    </p>
                  </div>
                  <div>
                    <label className="text-sm text-muted-foreground">Role</label>
                    <p className="text-foreground">
                      <span className={`px-2 py-1 rounded text-xs ${selectedUser.role === 'ADMIN' ? 'bg-purple-500/20 text-purple-500' : 'bg-blue-500/20 text-blue-500'}`}>
                        {selectedUser.role === 'ADMIN' ? 'Administrator' : 'Student (User)'}
                      </span>
                    </p>
                  </div>
                  {selectedUser.plain_password && (
                    <div>
                      <label className="text-sm text-muted-foreground">Current Password</label>
                      <p className="text-red-400 font-mono text-sm bg-red-950/30 p-1 rounded border border-red-900/50 inline-block">
                        {selectedUser.plain_password}
                      </p>
                    </div>
                  )}
                  {selectedUser.role === 'USER' && (
                    <div>
                      <label className="text-sm text-muted-foreground">Access Type</label>
                      <p className="text-foreground">
                        <span
                          className={`px-2 py-1 rounded text-xs ${selectedUser.access_type === 'FULL'
                            ? "bg-primary/20 text-primary font-medium"
                            : "bg-secondary/20 text-secondary"
                            }`}
                        >
                          {selectedUser.access_type === 'FULL' ? 'Full Access' : 'Trial'}
                        </span>
                      </p>
                    </div>
                  )}
                  {selectedUser.access_type === 'TRIAL' && (
                    <div>
                      <label className="text-sm text-muted-foreground">Trial Status</label>
                      <p className="text-foreground">
                        <span
                          className={`px-2 py-1 rounded text-xs ${selectedUser.has_used_trial
                            ? "bg-secondary/20 text-secondary"
                            : "bg-primary/20 text-primary"
                            }`}
                        >
                          {selectedUser.has_used_trial ? "Used" : "Available"}
                        </span>
                      </p>
                    </div>
                  )}
                  <div>
                    <label className="text-sm text-muted-foreground">Created</label>
                    <p className="text-foreground text-sm">
                      {new Date(selectedUser.created_at).toLocaleString()}
                    </p>
                  </div>
                </div>

                <div className="flex flex-col gap-4 mt-6">
                  {showPasswordChange ? (
                    <div className="p-4 rounded-lg bg-muted/20 border border-border">
                      <h4 className="text-sm font-medium text-foreground mb-3">Change Password</h4>
                      <div className="flex gap-2">
                        <Input
                          type="text"
                          placeholder="Enter new password"
                          value={newPassword}
                          onChange={(e) => setNewPassword(e.target.value)}
                          className="bg-input border-border"
                        />
                        <Button
                          onClick={handleChangePassword}
                          disabled={!newPassword || newPassword.length < 6}
                          className="whitespace-nowrap"
                        >
                          Save
                        </Button>
                        <Button
                          variant="ghost"
                          onClick={() => {
                            setShowPasswordChange(false);
                            setNewPassword("");
                          }}
                        >
                          Cancel
                        </Button>
                      </div>
                      <p className="text-xs text-muted-foreground mt-2">
                        Password must be at least 6 characters.
                      </p>
                    </div>
                  ) : null}

                  <div className="flex flex-wrap gap-2">
                    <Button
                      variant="outline"
                      size="sm"
                      onClick={() => handleToggleUserStatus(selectedUser)}
                    >
                      {selectedUser.is_active ? (
                        <>
                          <EyeOff className="h-4 w-4 mr-2" />
                          Deactivate User
                        </>
                      ) : (
                        <>
                          <Eye className="h-4 w-4 mr-2" />
                          Activate User
                        </>
                      )}
                    </Button>
                    {selectedUser.role === 'USER' && (
                      <Button
                        variant="outline"
                        size="sm"
                        onClick={handleUpdateAccessType}
                      >
                        {selectedUser.access_type === 'FULL' ? 'Set to Trial' : 'Set to Full Access'}
                      </Button>
                    )}
                    {!showPasswordChange && (
                      <Button
                        variant="outline"
                        size="sm"
                        onClick={() => setShowPasswordChange(true)}
                      >
                        <Lock className="h-4 w-4 mr-2" />
                        Change Password
                      </Button>
                    )}
                    {selectedUser.has_used_trial && (
                      <Button
                        variant="outline"
                        size="sm"
                        onClick={() => handleResetTrial(selectedUser.id)}
                      >
                        <RefreshCw className="h-4 w-4 mr-2" />
                        Reset Trial
                      </Button>
                    )}
                    <Button
                      variant="destructive"
                      size="sm"
                      onClick={() => handleDeleteUser(selectedUser.id)}
                      className="ml-auto"
                    >
                      <Trash2 className="h-4 w-4 mr-2" />
                      Delete User
                    </Button>
                  </div>
                </div>
              </Card>

              {/* Interview Sessions */}
              {/* Interview Sessions - Only for Students */}
              {selectedUser.role === 'USER' && (
                <Card className="p-4 bg-card/50 border-border">
                  <h3 className="text-lg font-semibold text-foreground mb-4">
                    Interview Sessions ({userSessions.length})
                  </h3>

                  {isLoadingSessions ? (
                    <div className="text-center py-8 text-muted-foreground">
                      Loading sessions...
                    </div>
                  ) : userSessions.length === 0 ? (
                    <div className="text-center py-8 text-muted-foreground">
                      No interview sessions found for this user.
                    </div>
                  ) : (
                    <div className="space-y-3">
                      {userSessions.map((session) => (
                        <div
                          key={session.id}
                          className="p-3 bg-muted/20 rounded-lg border border-border"
                        >
                          <div className="flex justify-between items-start mb-2">
                            <div>
                              <p className="font-medium text-foreground">
                                Session #{session.id}
                              </p>
                              <p className="text-sm text-muted-foreground">
                                {new Date(session.started_at).toLocaleString()}
                              </p>
                            </div>
                            <span
                              className={`px-2 py-1 rounded text-xs ${session.status === 'COMPLETED'
                                ? "bg-primary/20 text-primary"
                                : session.status === 'IN_PROGRESS'
                                  ? "bg-secondary/20 text-secondary"
                                  : "bg-muted text-muted-foreground"
                                }`}
                            >
                              {session.status}
                            </span>
                          </div>

                          {session.topics_list && session.topics_list.length > 0 && (
                            <div className="mb-2">
                              <p className="text-xs text-muted-foreground mb-1">Topics:</p>
                              <div className="flex flex-wrap gap-1">
                                {session.topics_list.map((topic: any) => (
                                  <span
                                    key={topic.id}
                                    className="px-2 py-0.5 rounded text-xs bg-primary/10 text-primary"
                                  >
                                    {topic.name}
                                  </span>
                                ))}
                              </div>
                            </div>
                          )}

                          {session.status === 'COMPLETED' && (
                            <div className="grid grid-cols-2 gap-2 mt-2">
                              <div>
                                <p className="text-xs text-muted-foreground">Communication Score</p>
                                <p className="text-foreground font-medium">
                                  {session.communication_score !== null
                                    ? `${(session.communication_score * 100).toFixed(1)}%`
                                    : "N/A"}
                                </p>
                              </div>
                              <div>
                                <p className="text-xs text-muted-foreground">Technology Score</p>
                                <p className="text-foreground font-medium">
                                  {session.technology_score !== null
                                    ? `${(session.technology_score * 100).toFixed(1)}%`
                                    : "N/A"}
                                </p>
                              </div>
                            </div>
                          )}
                        </div>
                      ))}
                    </div>
                  )}
                </Card>
              )}
            </div>
          )}
        </DialogContent>
      </Dialog>
    </div>
  );
};

export default AdminDashboard;
