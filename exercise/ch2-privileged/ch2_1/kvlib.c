#define _GNU_SOURCE
#include <lua.h>
#include <lauxlib.h>
#include <sys/syscall.h>
#include <unistd.h>
#include <stdio.h>

#define SYS_write_kv 449
#define SYS_read_kv  450

static int l_write_kv(lua_State *L) {
    int k = luaL_checkinteger(L, 1);
    int v = luaL_checkinteger(L, 2);
    printf("[C] Calling syscall write_kv(%d, %d)\n", k, v);
    int ret = syscall(SYS_write_kv, k, v);
    printf("[C] Returned from syscall: %d\n", ret);
    lua_pushinteger(L, ret);
    return 1;
}

static int l_read_kv(lua_State *L) {
    int k = luaL_checkinteger(L, 1);
    printf("[C] Calling syscall read_kv(%d)\n", k);
    int ret = syscall(SYS_read_kv, k);
    printf("[C] Returned from syscall: %d\n", ret);
    lua_pushinteger(L, ret);
    return 1;
}

static const struct luaL_Reg kvlib[] = {
    {"write_kv", l_write_kv},
    {"read_kv",  l_read_kv},
    {NULL, NULL}
};

int luaopen_kvlib(lua_State *L) {
    luaL_newlib(L, kvlib);
    return 1;
}
