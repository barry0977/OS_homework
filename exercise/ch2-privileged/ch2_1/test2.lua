-- Define the range of keys
local MAX_KEY = 2047
local kv = require("kvlib") -- 加载我的动态库
function thread_init(thread_id)
    -- Initialize per-thread variables
    math.randomseed(os.time() + thread_id)
end
function event(thread_id)
    local k = math.random(0, MAX_KEY)
    local v = math.random(0,1000)
    local operation = math.random(0, 1) -- 0 for read, 1 for write
    if operation == 1 then
        -- Write Operation
        if kv.write_kv(k, v) == -1 then
            sb_report_error("Error writing key " .. k)
        end
    else
        -- Read Operation
        local read_value = kv.read_kv(k)
        if read_value == -1 then
            sb_report_error("Error reading key " .. k)
        end
    end
end
-- sysbench --threads=100 --events=1000000 kv_test.lua run