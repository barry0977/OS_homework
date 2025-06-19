//用于给test 2 Lua C库
//gcc -Wall -fPIC -shared -o kvlib.so kvlib.c
#define _GNU_SOURCE
#include <lua5.3/lua.h>
#include <lua5.3/lauxlib.h>
#include <sys/syscall.h>
#include <unistd.h>

#define SYS_write_kv 449
#define SYS_read_kv  450

// Lua wrapper for write_kv(k, v)
static int l_write_kv(lua_State *L) {
    int k = luaL_checkinteger(L, 1);
    int v = luaL_checkinteger(L, 2);
    printf("write_kv called with k=%d, v=%d\n", k, v);
    int ret = syscall(SYS_write_kv, k, v);
    lua_pushinteger(L, ret);
    return 1;
}

// Lua wrapper for read_kv(k)
static int l_read_kv(lua_State *L) {
    int k = luaL_checkinteger(L, 1);
    int ret = syscall(SYS_read_kv, k);
    lua_pushinteger(L, ret);
    return 1;
}

// 注册函数
static const struct luaL_Reg kvlib[] = {
    {"write_kv", l_write_kv},
    {"read_kv",  l_read_kv},
    {NULL, NULL}
};

int luaopen_kvlib(lua_State *L) {
    luaL_newlib(L, kvlib);
    return 1;
}
