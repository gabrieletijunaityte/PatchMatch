"""
This file contains all SQL code queries that are used in database/ProjectDB.py
"""

CREATE_MATCH = """
                    CREATE TABLE IF NOT EXISTS "Match" 
                    (
                        "MatchID"	INTEGER NOT NULL UNIQUE,
                        "PatchID_PS"	INTEGER NOT NULL,
                        "PatchID_S2"	INTEGER NOT NULL,
                        "TestTrainValidate" INTEGER DEFAULT 0,
                        PRIMARY KEY("MatchID"),
                        FOREIGN KEY("PatchID_PS") REFERENCES "Patch"("PatchID"),
                        FOREIGN KEY("PatchID_S2") REFERENCES "Patch"("PatchID")
                    )
                """

CREATE_PATCH = """
                    CREATE TABLE IF NOT EXISTS "Patch" 
                    (
                        "PatchID"	INTEGER NOT NULL UNIQUE,
                        "SceneID"	TEXT NOT NULL,
                        "CentreLat"	REAL NOT NULL,
                        "CentreLon"	REAL NOT NULL,
                        PRIMARY KEY("PatchID"),
                        FOREIGN KEY("SceneID") REFERENCES "Scene"("SceneID")
                    )
"""

CREATE_SCENE = """
                    CREATE TABLE IF NOT EXISTS "Scene" 
                    (
                        "SceneID"	TEXT NOT NULL UNIQUE,
                        "StudySite"	TEXT NOT NULL,
                        "CRS"	TEXT NOT NULL,
                        "Platform"	TEXT CHECK("Platform" IN ('S2', 'PS')),
                        "Sensor"	TEXT NOT NULL,
                        "PxSize"	REAL,
                        "AcqDate"	TEXT NOT NULL,
                        "AcqTime"	TEXT NOT NULL,
                        "TopRightCornerLat"	REAL NOT NULL,
                        "TopRightCornerLon"	REAL NOT NULL,
                        "BottomLeftCornerLat"	REAL NOT NULL,
                        "BottomLeftCornerLon"	REAL NOT NULL,
                        PRIMARY KEY("SceneID")
                    )
"""

GET_PS = """    
                    SELECT SceneID 
                    FROM Scene 
                    WHERE Platform == "PS"
                """

GET_S2 = """    
                    SELECT SceneID 
                    FROM Scene 
                    WHERE Platform == "S2"
        """

GET_SITES = """
                    SELECT SceneID 
                    FROM Scene
            """

GET_SCENES = """
                    SELECT DISTINCT StudySite 
                    FROM Scene
            """

GET_MATCH_IDS = """
                    SELECT MatchID
                    FROM Match
                """

GET_TRAIN_TEST_MATCHES = """
                    SELECT Match.MatchID
                    FROM Match
                    WHERE Match.TestTrainValidate=?;
"""

GET_SITE_DATE = """
                    SELECT min(AcqDate) 
                    FROM Scene 
                    WHERE StudySite = ?
                """

GET_S2_FROM_PS = """
                    SELECT SceneID 
                    FROM Scene 
                    WHERE StudySite == (
                                        SELECT StudySite 
                                        FROM Scene 
                                        WHERE SceneID == ?) 
                    AND Platform == 'S2'              
                """

GET_SITE_CRS = """
                    SELECT DISTINCT crs 
                    FROM Scene 
                    WHERE StudySite == ? 
                    AND Platform == ?
                """

GET_STUDY_SITE = """
                    SELECT StudySite
                    FROM Scene 
                    WHERE SceneID == ?
                """

GET_PLATFORM = """
                    SELECT platform 
                    FROM Scene 
                    WHERE SceneID == ?
                """

GET_COMBINED_BBOX = """
                    SELECT 
                        max(TopRightCornerLat), max(TopRightCornerLon), 
                        min(BottomLeftCornerLat), min(BottomLeftCornerLon) 
                    FROM Scene 
                    WHERE 
                        StudySite == ? 
                        AND
                        platform == ?
                """

GET_MATCH_INFO = """
                    SELECT 
                        PatchID_PS, 
                        p1.CentreLat AS CentreLat_PS, 
                        p1.CentreLon AS CentreLon_PS, 
                        p1.SceneID as PS_SceneID,
                        PatchID_S2, 
                        p2.CentreLat AS CentreLat_S2, 
                        p2.CentreLon AS CentreLon_S2,
                        p2.SceneID as S2_SceneID
                    FROM Match
                    JOIN Patch p1 ON PatchID_PS = p1.PatchID
                    JOIN Patch p2 ON PatchID_S2 = p2.PatchID
                    WHERE MatchID = ?;
                """

ADD_NEW_SCENE = """
                    INSERT INTO Scene
                        (SceneID, Platform, 
                        Sensor, PxSize, 
                        AcqDate, AcqTime,
                        TopRightCornerLat, TopRightCornerLon,
                        BottomLeftCornerLat, BottomLeftCornerLon, 
                        StudySite, CRS)
                    VALUES (?,?,?,?,?,?,?,?,?,?,?,?)
                """

UPDATE_TEST_SITES = """
                    UPDATE Match
                    SET TestTrainValidate = CASE
                        WHEN PatchID_S2 == ? THEN ?
                        ELSE TestTrainValidate
                    END;
                """

ADD_NEW_PATCH = """
                    INSERT OR IGNORE INTO Patch
                        (SceneID, CentreLat, CentreLon)
                    VALUES (?,?,?)
                """

GET_NEW_PATCH_ID = """
                    SELECT PatchID 
                    FROM Patch 
                    ORDER BY PatchID DESC 
                    LIMIT ?
                    """

ADD_NEW_MATCH = """
                    INSERT INTO Match 
                        (PatchID_PS, PatchID_S2)
                    VALUES (?, ?)
                """

# https://www.atlassian.com/data/sql/how-to-find-duplicate-values-in-a-sql-table
GET_DUPLICATE_PATCH_IDS = """
                    SELECT 
                        GROUP_CONCAT(PatchID) as DuplicateIDs
                    FROM Patch
                    GROUP BY SceneID, CentreLat, CentreLon
                    HAVING COUNT(*) > 1
                    """

# https://stackoverflow.com/questions/14686204/how-to-replace-sql-field-value
REPLACE_DUPLICATE_PS_IDS = """
                    UPDATE Match
                    SET PatchID_PS = ?
                    WHERE PatchID_PS IN ({});

                        """

REPLACE_DUPLICATE_S2_IDS = """
                    UPDATE Match
                    SET PatchID_S2 = ?
                    WHERE PatchID_S2 IN ({})
                        """

DELETE_PATCH_IDS = """
                    DELETE FROM Patch
                    WHERE PatchID IN ({})
                    """

GET_DUPLICATE_MATCH_IDS = """
                    SELECT 
                        GROUP_CONCAT(MatchID) as DuplicateIDs
                    FROM Match
                    GROUP BY PatchID_PS, PatchID_S2
                    HAVING COUNT(*) > 1
                    """

DELETE_MATCH_IDS = """
                    DELETE FROM Match
                    WHERE MatchID IN ({})
                    """

GET_S2_PATCHES = """
                    SELECT 
                        Patch.PatchID, Patch.SceneID, Patch.CentreLat, Patch.CentreLon, Scene.StudySite
                    FROM Patch 
                    JOIN Scene ON Patch.SceneID=Scene.SceneID 
                    WHERE Scene.Platform="S2"
"""

GET_S2_COORD_STUDYSITE = """
                    SELECT 
                        Patch.PatchID, Patch.CentreLat, Patch.CentreLon 
                    FROM Patch 
                    JOIN Scene ON Patch.SceneID=Scene.SceneID 
                    WHERE Scene.StudySite=? AND Scene.Platform=?
"""

GET_ID_STUDYSITE_TEST = """
                    SELECT 
                        PatchID 
                    FROM Patch 
                    JOIN Scene ON Patch.SceneID=Scene.SceneID
                    JOIN Match ON Patch.PatchID=Match.PatchID_X
                    WHERE Match.TestTrainValidate==1 AND Scene.StudySite=?
"""

GET_MATCH_STATUS = """
                        SELECT 
                            ps.CentreLat AS PS_Lat, 
                            ps.CentreLon AS PS_Lon, 
                            s2.CentreLat AS S2_Lat, 
                            s2.CentreLon AS S2_Lon,
                            CASE 
                                WHEN s2.PatchID IS NOT NULL AND EXISTS (
                                    SELECT 1 FROM Match 
                                    WHERE PatchID_PS = ps.PatchID AND PatchID_S2 = s2.PatchID
                                ) THEN 1
                                ELSE 0
                            END AS Match
                        FROM Patch ps
                        LEFT JOIN Patch s2 ON s2.PatchID = ?  
                        WHERE ps.PatchID = ?;
"""

GET_TIME_DIFFERENCES = """
                        SELECT 
                            Match.PatchID_PS, 
                            p1.CentreLat, 
                            p1.CentreLon,
                            Match.PatchID_S2, 
                            ABS(strftime('%s', s2.AcqTime) - strftime('%s', s1.AcqTime)) AS TimeDiffInSeconds
                        FROM Match
                        JOIN Patch p1 ON Match.PatchID_PS = p1.PatchID
                        JOIN Patch p2 ON Match.PatchID_S2 = p2.PatchID
                        JOIN Scene s1 ON p1.SceneID = s1.SceneID
                        JOIN Scene s2 ON p2.SceneID = s2.SceneID
                        WHERE TestTrainValidate = ? AND s2.StudySite = ?;
                        """

GET_POINT_WITHIN_RADIUS = """
                        SELECT DISTINCT
                            Patch.PatchID
                        FROM Patch
                        JOIN Scene  ON Patch.SceneID = Scene.SceneID
                        JOIN Match ON Match.PatchID_S2=Patch.PatchID
                        WHERE Match.TestTrainValidate = ? 
                            AND Scene.StudySite=? 
                            AND Scene.Platform=?
                            AND  sqrt((Patch.CentreLat - ?) * (Patch.CentreLat - ?) +   (Patch.CentreLon - ?) * (Patch.CentreLon - ?) ) <= ?;
"""