from read_sheet.record_gs import Record

class Main:
    def fetch_project(self):
        """start input request for user_name and project
        objects:
            Record_gs
            Record_csv
            Record_direct
        types:
            (google sheets): read project info via google spreadsheets
            (csv): read project info via filename.csv
            (direct): manual input requests for project info

        Args:
            user_name (str): name of the user
            project_general (str): project name / start / end / team
            project_insights (csv): task / member assigned
        """
