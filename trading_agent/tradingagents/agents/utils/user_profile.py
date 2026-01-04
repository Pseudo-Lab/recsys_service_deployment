
import os
import json
from typing import List, Optional
from pydantic import BaseModel, Field
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from threading import Lock

class UserProfile(BaseModel):
    risk_tolerance: str = Field(description="Risk tolerance level: 'Conservative', 'Moderate', 'Aggressive', 'Speculative'")
    investment_horizon: str = Field(description="Investment time horizon: 'Short-term', 'Mid-term', 'Long-term'")
    favorite_sectors: List[str] = Field(description="List of sectors the user is interested in")
    avoid_sectors: List[str] = Field(description="List of sectors the user wants to avoid")
    investment_style: str = Field(description="Brief description of investment style (e.g. 'Value Investing', 'Growth', 'Dividend')")
    custom_instructions: str = Field(description="Specific instructions or constraints for the agents")

    @property
    def summary(self) -> str:
        return (
            f"Risk Tolerance: {self.risk_tolerance}\n"
            f"Horizon: {self.investment_horizon}\n"
            f"Style: {self.investment_style}\n"
            f"Favorites: {', '.join(self.favorite_sectors)}\n"
            f"Avoid: {', '.join(self.avoid_sectors)}\n"
            f"Instructions: {self.custom_instructions}"
        )

DEFAULT_PROFILE = UserProfile(
    risk_tolerance="Moderate",
    investment_horizon="Mid-term",
    favorite_sectors=[],
    avoid_sectors=[],
    investment_style="Balanced",
    custom_instructions="Focus on fundamental analysis and market trends."
)

class UserProfileManager:
    def __init__(self, data_dir: str = "./dataflows/data_cache", llm_model: str = "gpt-4o-mini"):
        self.profile_path = os.path.join(data_dir, "user_profile.json")
        os.makedirs(data_dir, exist_ok=True)
        self.lock = Lock()
        
        # Initialize LLM for profile extraction
        self.llm = ChatOpenAI(model=llm_model, temperature=0)
        self.parser = JsonOutputParser(pydantic_object=UserProfile)
        
        self.prompt = ChatPromptTemplate.from_messages([
            ("system", "You are an expert Investment Profile Manager. Your job is to update the user's investment profile based on their chat messages.\n"
                       "You will be given the CURRENT PROFILE and the NEW USER INPUT.\n"
                       "Update the fields that the user explicitly mentions or implies changes to. Keep other fields unchanged.\n"
                       "If the user says 'I like tech', add 'Technology' to favorite_sectors.\n"
                       "If the user says 'I am aggressive', set risk_tolerance to 'Aggressive'.\n"
                       "Format your output as a valid JSON matching the UserProfile schema.\n"
                       "{format_instructions}"),
            ("user", "CURRENT PROFILE:\n{current_profile}\n\nUSER INPUT: {user_input}")
        ])

    def load_profile(self) -> UserProfile:
        with self.lock:
            if not os.path.exists(self.profile_path):
                return DEFAULT_PROFILE.model_copy()
            try:
                with open(self.profile_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                return UserProfile(**data)
            except Exception as e:
                print(f"Error loading profile: {e}, returning default.")
                return DEFAULT_PROFILE.model_copy()

    def save_profile(self, profile: UserProfile):
        with self.lock:
            with open(self.profile_path, "w", encoding="utf-8") as f:
                json.dump(profile.model_dump(), f, indent=2, ensure_ascii=False)

    def update_profile_from_text(self, user_input: str) -> UserProfile:
        current_profile = self.load_profile()
        
        chain = self.prompt | self.llm | self.parser
        
        try:
            print(f"DEBUG: Extracting profile from: {user_input}")
            new_data = chain.invoke({
                "current_profile": current_profile.model_dump_json(),
                "user_input": user_input,
                "format_instructions": self.parser.get_format_instructions()
            })
            
            # Merge logic is handled by LLM effectively regenerating the whole object, 
            # but we sanity check the result
            updated_profile = UserProfile(**new_data)
            self.save_profile(updated_profile)
            print(f"DEBUG: Profile updated: {updated_profile.summary}")
            return updated_profile
            
        except Exception as e:
            print(f"ERROR updating profile: {e}")
            return current_profile

# Global instance for easy access
profile_manager = UserProfileManager()
