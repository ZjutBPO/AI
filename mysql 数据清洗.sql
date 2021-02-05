-- 更新进站口所属地铁线
UPDATE total
SET InLine =
(SELECT LEFT(SubwayLine,1)
FROM station
WHERE StationName = total.InStationName)

-- 更新出站口所属地铁线
UPDATE total
SET OutLine =
(SELECT LEFT(SubwayLine,1)
FROM station
WHERE StationName = total.OutStationName)

-- 查询进站日期和出站日期不是同一天的记录。即横跨午夜12点
Select * from 
total
Where LEFT(InTime,10) <> LEFT(OutTime,10)

-- 按照进站日期给每条记录标注所属日期类型
UPDATE total
SET DateType = (Select type
from workdays2020
Where date = CONVERT ( InTime, date )


-- 删除所属地铁线路为NULL的地铁站（60000+条）
DELETE
FROM total
WHERE InLine is NULL or OutLine is NULL

-- 删除金额不是整百的记录（27条）
DELETE
FROM total
WHERE Price % 100 <> 0

-- 删除2019年的数据(3433条)
DELETE 
FROM total
WHERE DateType is NULL