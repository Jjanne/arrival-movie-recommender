import { auth } from "@/lib/firebase";
import type { ChatMessage } from "@/types/chat";

type HistoryItem = { role: "user" | "assistant"; content: string };

/**
 * Stream tokens from the backend SSE endpoint.
 *
 * Yields each text token as it arrives. Throws on non-2xx responses or
 * on a server-sent error event.
 *
 * @param message  The current user turn.
 * @param history  Prior conversation turns (role + content only).
 */
export async function* streamChatMessage(
  message: string,
  history: HistoryItem[],
): AsyncGenerator<string> {
  const user = auth.currentUser;
  const idToken = user ? await user.getIdToken() : null;

  const baseUrl = (import.meta.env.VITE_BASE_URL ?? "").replace(/\/$/, "");

  const response = await fetch(`${baseUrl}/api/v1/chat/stream`, {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
      ...(idToken ? { Authorization: `Bearer ${idToken}` } : {}),
    },
    body: JSON.stringify({ message, history }),
  });

  if (!response.ok) {
    throw new Error(`Chat stream request failed: ${response.status}`);
  }

  const reader = response.body!.getReader();
  const decoder = new TextDecoder();
  let buffer = "";

  while (true) {
    const { done, value } = await reader.read();
    if (done) break;

    buffer += decoder.decode(value, { stream: true });

    // SSE lines are separated by \n; events by \n\n.
    // Keep any trailing incomplete line in the buffer for the next chunk.
    const lines = buffer.split("\n");
    buffer = lines.pop() ?? "";

    for (const line of lines) {
      const trimmed = line.trimEnd();
      if (!trimmed.startsWith("data:")) continue;

      const data = trimmed.slice(5).trim();
      if (data === "[DONE]") return;

      let parsed: Record<string, unknown>;
      try {
        parsed = JSON.parse(data);
      } catch {
        continue; // malformed line — skip
      }

      if (typeof parsed.error === "string") {
        throw new Error(parsed.error);
      }
      if (typeof parsed.token === "string" && parsed.token.length > 0) {
        yield parsed.token;
      }
    }
  }
}

export async function getChatHistory(): Promise<ChatMessage[]> {
  return [];
}
