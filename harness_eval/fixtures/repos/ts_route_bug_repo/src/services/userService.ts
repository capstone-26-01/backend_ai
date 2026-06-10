type UserInput = {
  email?: string;
};

export async function createUser(input: UserInput) {
  return {
    id: "user_123",
    email: input.email!.trim().toLowerCase(),
  };
}

