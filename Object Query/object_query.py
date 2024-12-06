# helpful video: https://www.youtube.com/watch?v=lJJkBaO15Po
class ObjectQuery:
    def __init__(self, environment_object="object"):
        """Initialize the ObjectQuery with an environment object."""
        self.environment_object = environment_object

    def set_environment_object(self, new_object):
        """Set a new environment object to query about."""
        self.environment_object = new_object

    def generate_prompt(self):
        """Generate a prompt for the specified environment object."""
        llm_prompt = f"""
       Please respond with a single word: high, medium, or low. How dangerous would it be for a free-moving robot to collide with a {self.environment_object}? 
       Typically, classify living, fragile, electronics, or dangerous objects as 'high,' while most other objects fall under 'medium' or 'low.' Be conservative in your analysis.
        """
        return llm_prompt
    
    def generate_number_prompt(self):
        """Generate a prompt for the specified environment object."""
        llm_prompt = f"""
       Only give me a single natural number as your response, On a scale of 1-10 (10 being severe) How dangerous would it be for a free moving robot to colide with a {self.environment_object}? 
       Typically, classify living, fragile, electronics, or dangerous objects as severe while most other objects are not as severe. Be conservative in your analysis.
        """
        return llm_prompt

    
    @staticmethod
    def get_Severity(response):
        if response == "high":
            severity_level = 10
        elif response == "medium":
            severity_level = 5
        elif response == "low":
            severity_level = 1
        else: 
            severity_level = 7
        return severity_level