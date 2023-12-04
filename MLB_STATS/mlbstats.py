import requests
import mlbstatsapi
from datetime import datetime, timedelta
import psycopg2
import psycopg2.extras


MLB = mlbstatsapi.Mlb()
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
            print("Get {} stats at {}".format(key, date))
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
            print("Get {} stats at {}".format(key, date))
            stat = get_team_stats(team_id, date)
            if stat is None:
                continue
            if key not in full_data_2023:
                full_data_2023[key] = {}
            full_data_2023[key][date] = stat

    return full_data_2023
  

def insert_data():
    full_data_2022 = get_all_data_2022()
    full_data_2023 = get_all_data_2023()

    print("Data collection done. ")

    conn = psycopg2.connect(host='localhost',
                            dbname='mlbstats',
                            user='postgres',
                            password='Ghdtpauddlek1@',
                            port=5432
    )

    cursor = conn.cursor()

    for team in full_data_2022:
        for date in full_data_2022[team]:
            try:
                query = build_insert_query(full_data_2022[team][date], team, date)
                cursor.execute(query)
                conn.commit()
            except Exception as e:
                print(e)
    
    for team in full_data_2023:
        for date in full_data_2023[team]:
            try:
                query = build_insert_query(full_data_2023[team][date], team, date)
                cursor.execute(query)
                conn.commit()
            except Exception as e:
                print(e)
    
    conn.close()


def build_insert_query(team_date_stats, team, date):
    avg = str(team_date_stats['avg'])
    ops = str(team_date_stats['ops'])
    era = str(team_date_stats['era'])
    whip = str(team_date_stats['whip'])
    runs = str(team_date_stats['runs'])
    earnedRuns = str(team_date_stats['earnedRuns'])
    insert = "INSERT INTO mlbstats.dateteamstats"
    columns = "(team_name, avg, ops, era, whip, runs, earnedruns, date)"
    values = " VALUES('" + team + "', " + avg + ", " + ops + ", " + era + ", "
    values_2 = whip + ", " + runs + ", " + earnedRuns + ", '" + date + "');"
    sql = insert + columns + values + values_2

    return sql


def get_schedule():
    global START_2022
    global END_2022
    global START_2023
    global END_2023
    date_range_1 = get_date_range(START_2022, END_2022)
    date_range_2 = get_date_range(START_2023, END_2023)
    full_date_range = date_range_1 + date_range_2
    data_lst = []
    for date in full_date_range:
        url = 'https://statsapi.mlb.com/api/v1/schedule/games/?sportId=1&date='
        full_url = url + date
        response = requests.get(full_url)
        data = response.json()
        try:
            num_total_games = int(data['totalGames'])
        except Exception as e:
            print(e)
        for i in range(0, num_total_games):
            try:
                game_result = {}
                games = data['dates'][0]['games'][i]['teams']
                away = games['away']
                home = games['home']
                game_result['date'] = date
                game_result['home_team'] = home['team']['name']
                game_result['away_team'] = away['team']['name']
                print("Get {} vs {} at {}".format(home['team']['name'], away['team']['name'], date))
                game_result['home_team_score'] = home['score']
                game_result['away_team_score'] = away['score']
                if home['isWinner'] == True or home['isWinner'] == 'True':
                    game_result['winner'] = 1
                else:
                    game_result['winner'] = 0
                
                data_lst.append(game_result)
            except Exception as e:
                print(e)
        
    return data_lst
        

def insert_schedule():
    schedule = get_schedule()
    print()
    print("Start INSERT")
    conn = psycopg2.connect(host='localhost',
                            dbname='mlbstats',
                            user='postgres',
                            password='Ghdtpauddlek1@',
                            port=5432
    )

    cursor = conn.cursor()

    for game_result in schedule:
        try:
            query = build_schedule_query(game_result)
            cursor.execute(query)
            conn.commit()
        except Exception as e:
            print(e)
    
    conn.close()


def build_schedule_query(game_result):
    date = str(game_result['date'])
    home_team = str(game_result['home_team'])
    away_team = str(game_result['away_team'])
    home_team_score = str(game_result['home_team_score'])
    away_team_score = str(game_result['away_team_score'])
    winner = str(game_result['winner'])
    insert = "INSERT INTO mlbstats.gameresults"
    columns = "(date, home, away, home_score, away_score, winner)"
    values = " VALUES('" + date + "', '" + home_team + "', '" + away_team + "', "
    values_2 = home_team_score + ", " + away_team_score + ", " + winner + ");"
    sql = insert + columns + values + values_2

    return sql


def join_data():
    conn = psycopg2.connect(host='localhost',
                            dbname='mlbstats',
                            user='postgres',
                            password='Ghdtpauddlek1@',
                            port=5432
    )

    cursor = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
    query = "SELECT * FROM mlbstats.gameresults;"
    cursor.execute(query)
    rows = cursor.fetchall()

    full_data = []
    for row in rows:
        date = row['date']
        home = row['home']
        away = row['away']
        try:
            query_home = "SELECT * FROM mlbstats.dateteamstats WHERE date = '{}' AND team_name = '{}';".format(date, home)
            cursor.execute(query_home)
            home_stats = cursor.fetchall()[0]
            query_away = "SELECT * FROM mlbstats.dateteamstats WHERE date = '{}' AND team_name = '{}';".format(date, away)
            cursor.execute(query_away)
            away_stats = cursor.fetchall()[0]
            full_input = {}
            for key in home_stats:
                if key != 'date':
                    full_input[key+'_home'] = home_stats[key]
            for key in away_stats:
                if key != 'date':
                    full_input[key+'_away'] = away_stats[key]
            full_input['date'] = datetime.strftime(date, "%m/%d/%Y")
            full_input['target'] = row['winner']
            full_data.append(full_input)
        except Exception as e:
            print(e)

    return full_data


def insert_input_target():
    full_data = join_data()
    print()
    print("Start INSERT")
    conn = psycopg2.connect(host='localhost',
                            dbname='mlbstats',
                            user='postgres',
                            password='Ghdtpauddlek1@',
                            port=5432
    )

    cursor = conn.cursor()

    for input in full_data:
        try:
            query = build_input_query(input)
            cursor.execute(query)
            conn.commit()
        except Exception as e:
            print(e)
    
    conn.close()


def build_input_query(input):
    team_name_home = str(input['team_name_home'])
    avg_home = str(input['avg_home'])
    ops_home = str(input['ops_home'])
    era_home = str(input['era_home'])
    whip_home = str(input['whip_home'])
    runs_home = str(input['runs_home'])
    earnedruns_home = str(input['earnedruns_home'])
    team_name_away = str(input['team_name_away'])
    avg_away = str(input['avg_away'])
    ops_away = str(input['ops_away'])
    era_away = str(input['era_away'])
    whip_away = str(input['whip_away'])
    runs_away = str(input['runs_away'])
    earnedruns_away = str(input['earnedruns_away'])
    date = str(input['date'])
    target = str(input['target'])

    insert = "INSERT INTO mlbstats.inputs"
    columns = "(home, avg_home, ops_home, era_home, whip_home, runs_home, earnedruns_home, away, avg_away, ops_away, era_away, whip_away, runs_away, earnedruns_away, date, target)"
    values = " VALUES('" + team_name_home + "', " + avg_home + ", " + ops_home + ", " + era_home + ", " + whip_home + ", " + runs_home + ", " + earnedruns_home + ", '"
    values_2 = team_name_away + "', " + avg_away + ", " + ops_away + ", " + era_away + ", " + whip_away + ", " + runs_away + ", " + earnedruns_away + ", '" + date + "', " + target + ");"
    sql = insert + columns + values + values_2

    return sql


if __name__ == '__main__':
    insert_data()
