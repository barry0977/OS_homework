#include <linux/module.h>
#include <linux/init.h>
#include <linux/acpi.h>
#include <linux/kobject.h>
#include <linux/sysfs.h>

#define ACPI_SIG_CUST "CUST" 

struct acpi_cust_table {
	struct acpi_table_header header;
	u32 version;
	u32 cpu_count;
	u64 timestamp;
	u32 what;
} __packed;

static struct acpi_cust_table *cust_table;
static struct kobject *cust_kobj;

//sysfs属性展示函数, 当用户读取对应文件时调用
static ssize_t version_show(struct kobject *kobj, struct kobj_attribute *attr, char *buf)
{
	return sprintf(buf, "%u\n", cust_table ? cust_table->version : 0);
}

static ssize_t cpu_count_show(struct kobject *kobj, struct kobj_attribute *attr, char *buf)
{
	return sprintf(buf, "%u\n", cust_table ? cust_table->cpu_count : 0);
}

static ssize_t timestamp_show(struct kobject *kobj, struct kobj_attribute *attr, char *buf)
{
	return sprintf(buf, "%llu\n", cust_table ? cust_table->timestamp : 0ULL);
}

//定义sysfs属性
static struct kobj_attribute version_attr = __ATTR_RO(version);
static struct kobj_attribute cpu_count_attr = __ATTR_RO(cpu_count);
static struct kobj_attribute timestamp_attr = __ATTR_RO(timestamp);

//定义属性数组
static struct attribute *cust_attrs[] = {
	&version_attr.attr,
	&cpu_count_attr.attr,
	&timestamp_attr.attr,
	NULL,
};

static struct attribute_group cust_attr_group = {
	.attrs = cust_attrs,
};

//解析自定义ACPI表 
static int parse_cust_acpi_table(void)
{
	struct acpi_table_header *table;
	acpi_status status;

	status = acpi_get_table(ACPI_SIG_CUST, 0, &table);
	if (ACPI_FAILURE(status)) {
		pr_warn("Custom ACPI table not found\n");
		return -ENODEV;
	}

	cust_table = (struct acpi_cust_table *)table;

	pr_info("Custom ACPI table found:\n");
	pr_info("  Version: %u\n", cust_table->version);
	pr_info("  CPU Count: %u\n", cust_table->cpu_count);
	pr_info("  Timestamp: 0x%llx\n", cust_table->timestamp);

	return 0;
}

static int __init cust_acpi_init(void)
{
	int ret;

	ret = parse_cust_acpi_table();
	if (ret)
		return ret;

	//创建sysfs目录 /sys/kernel/cust_acpi_info
	cust_kobj = kobject_create_and_add("cust_acpi_info", kernel_kobj);
	if (!cust_kobj) {
		pr_err("Failed to create cust_acpi_info kobject\n");
		return -ENOMEM;
	}

	ret = sysfs_create_group(cust_kobj, &cust_attr_group);
	if (ret) {
		kobject_put(cust_kobj);
		pr_err("Failed to create sysfs group\n");
		return ret;
	}

	return 0;
}

static void __exit cust_acpi_exit(void)
{
	if (cust_kobj)
		kobject_put(cust_kobj);
}

module_init(cust_acpi_init);
module_exit(cust_acpi_exit);

MODULE_LICENSE("GPL");
MODULE_AUTHOR("czx");
MODULE_DESCRIPTION("Custom ACPI Table Parser and Sysfs Exporter");
