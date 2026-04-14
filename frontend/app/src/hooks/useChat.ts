import { useChatStore } from "@/stores/chatStore";
import { streamChatMessage } from "@/services/api/chat";
import type { ChatMessage } from "@/types/chat";

export function useChat() {
  const { messages, isTyping, addMessage, appendToLastMessage, setTyping } =
    useChatStore();

  const sendMessage = async (content: string) => {
    // Capture prior turns before adding the new user message.
    // The backend receives these as `history` and the new turn as `message`.
    const history = messages.map((m) => ({ role: m.role, content: m.content }));

    addMessage({
      id: `msg-${Date.now()}`,
      content,
      role: "user",
      timestamp: new Date().toISOString(),
    });

    // Show the bouncing-dots indicator until the first token arrives.
    setTyping(true);

    let assistantMessageStarted = false;

    try {
      for await (const token of streamChatMessage(content, history)) {
        if (!assistantMessageStarted) {
          // First token: swap dots for the real message.
          setTyping(false);
          addMessage({
            id: `msg-${Date.now()}-ai`,
            content: token,
            role: "assistant",
            timestamp: new Date().toISOString(),
          });
          assistantMessageStarted = true;
        } else {
          appendToLastMessage(token);
        }
      }

      if (!assistantMessageStarted) {
        // Stream completed with no tokens (edge case).
        addMessage({
          id: `msg-${Date.now()}-ai`,
          content: "I couldn't generate a response. Please try again.",
          role: "assistant",
          timestamp: new Date().toISOString(),
        });
      }
    } catch {
      setTyping(false);
      if (!assistantMessageStarted) {
        addMessage({
          id: `msg-err-${Date.now()}`,
          content: "Sorry, I couldn't process your message. Please try again.",
          role: "assistant",
          timestamp: new Date().toISOString(),
        });
      }
    } finally {
      setTyping(false);
    }
  };

  return { messages, isTyping, sendMessage };
}
