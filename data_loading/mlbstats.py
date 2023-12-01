import requests
import statsapi
from datetime import datetime, timedelta


MLB = statsapi.Mlb()
START_2022 = "04/07/2022"
END_2022 = "10/02/2022"
START_2023 = "03/30/2023"
END_2023 = "10/01/2023"


def get_team_ids():
    global MLB
    all_teams = MLB.get_teams()
    id_dict = {}
    for team in all_teams:
        id_dict[team.name] = team.id
        
    return id_dict


def get_team_stats(team_id, date):
    end_point = "https://statsapi.mlb.com/api/v1/teams/"
    params = "/stats?group=pitching&season="
    params_2 = "&sportIds=1&stats=byDateRange&startDate=01/01/"
    params_3 = "&endDate="
    season = date[-4:]
    full_url = end_point + team_id + params + season + params_2 +season + params_3 + date

    response = requests.get(full_url)
    data = response.json()
    try:
        team_stat = data['stats'][0]['splits'][0]['stat']
    except Exception as e:
        print(e)
        return None

    data_dict = {}
    desired_stats = ['avg', 'ops', 'era', 'whip', 'runs', 'earnedRuns']
    for desired_stat in desired_stats:
        data_dict[desired_stat] = team_stat[desired_stat]
    
    return data_dict


def get_date_range(start_date, end_date):
    start = datetime.strptime(start_date, "%m/%d/%Y")
    end = datetime.strptime(end_date, "%m/%d/%Y")

    date_list = []

    current_date = start
    while current_date <= end:
        date_list.append(current_date.strftime("%m/%d/%Y"))
        current_date += timedelta(days=1)

    return date_list


def get_all_data_2022():
    global START_2022
    global END_2022
    date_range = get_date_range(START_2022, END_2022)
    team_ids = get_team_ids()

    full_data_2022 = {}
    for key in team_ids:
        team_id = str(team_ids[key])
        for date in date_range:
            stat = get_team_stats(team_id, date)
            if stat is None:
                continue
            if key not in full_data_2022:
                full_data_2022[key] = {}
            full_data_2022[key][date] = stat
    
    return full_data_2022
    
    
def get_all_data_2023():
    global START_2023
    global END_2023
    date_range = get_date_range(START_2023, END_2023)
    team_ids = get_team_ids()

    full_data_2023 = {}
    for key in team_ids:
        team_id = str(team_ids[key])
        for date in date_range:
            stat = get_team_stats(team_id, date)
            if stat is None:
                continue
            if key not in full_data_2023:
                full_data_2023[key] = {}
            full_data_2023[key][date] = stat

    return full_data_2023
