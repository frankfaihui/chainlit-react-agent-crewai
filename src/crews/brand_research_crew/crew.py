from crewai import Agent, Crew, Process, Task
from crewai.project import CrewBase, agent, crew, task

# Uncomment the following line to use an example of a custom tool
# from crewai_enterprise_content_marketing_ideas.tools.custom_tool import MyCustomTool

# Check our tools documentations for more information on how to use them
from crewai_tools import SerperDevTool


@CrewBase
class BrandResearchCrew:
    """BrandResearchCrew crew"""

    agents_config = "config/agents.yaml"
    tasks_config = "config/tasks.yaml"

    @agent
    def brand_research_agent(self) -> Agent:
        return Agent(
            config=self.agents_config["brand_research_agent"],
            tools=[SerperDevTool()],
            verbose=True,
        )

    @agent
    def strategy_agent(self) -> Agent:
        return Agent(config=self.agents_config["strategy_agent"], verbose=True)

    @task
    def brand_research_task(self) -> Task:
        return Task(
            config=self.tasks_config["brand_research_task"],
        )

    @task
    def strategy_task(self) -> Task:
        return Task(
            config=self.tasks_config["strategy_task"]
        )

    @crew
    def crew(self) -> Crew:
        """Creates the BrandResearchCrew"""
        return Crew(
            agents=self.agents,  # Automatically created by the @agent decorator
            tasks=self.tasks,  # Automatically created by the @task decorator
            process=Process.sequential,
            verbose=True,
            # process=Process.hierarchical, # In case you wanna use that instead https://docs.crewai.com/how-to/Hierarchical/
        )
