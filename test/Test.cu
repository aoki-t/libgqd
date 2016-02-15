#include "Test.h"

union trans{
	unsigned __int64 asInt64;
	double asDouble;
};

__device__
Test::Test() {

}

__device__
Test::~Test() {

}


__device__
void Test::expect_eq(double e, double a, char *ex, char *ac) {
	trans te, ta;
	te.asDouble = e;
	ta.asDouble = a;
	if (te.asInt64 == ta.asInt64) {
		printf("[ Succeeded ]  (%s == %s) expect:%f [0x%016llx], actual:%f [0x%016llx]\n", ex, ac, e, e, a, a);
	} else {
		printf("[ Failed    ]  (%s == %s) expect:%f [0x%016llx], actual:%f [0x%016llx]\n", ex, ac, e, e, a, a);
	}
}

__device__
void Test::expect_eq(unsigned __int64 e, double a, char *ex, char *ac) {
	trans ta;
	ta.asDouble = a;
	if (e == ta.asInt64) {
		printf("[ Succeeded ]  (%s == %s) expect:%f [0x%016llx], actual:%f [0x%016llx]\n", ex, ac, e, e, a, a);
	} else {
		printf("[ Failed    ]  (%s == %s) expect:%f [0x%016llx], actual:%f [0x%016llx]\n", ex, ac, e, e, a, a);
	}
}

__device__
void Test::expect_eq(bool e, bool a, char *ex, char *ac) {
	if (e == a) {
		printf("[ Succeeded ]  (%s == %s) expect:%s, actual:%s\n", ex, ac, (e ? "True" : "False"), (a ? "True" : "False"));
	} else {
		printf("[ Failed    ]  (%s == %s) expect:%s, actual:%s\n", ex, ac, (e ? "True" : "False"), (a ? "True" : "False"));
	}
}

__device__
void Test::expect_ne(double e, double a, char *ex, char *ac) {
	trans te, ta;
	te.asDouble = e;
	ta.asDouble = a;
	if (te.asInt64 != ta.asInt64) {
		printf("[ Succeeded ]  (%s != %s) expect:%f [0x%016llx], actual:%f [0x%016llx]\n", ex, ac, e, e, a, a);
	} else {
		printf("[ Failed    ]  (%s != %s) expect:%f [0x%016llx], actual:%f [0x%016llx]\n", ex, ac, e, e, a, a);
	}
}

__device__
void Test::expect_ne(unsigned __int64 e, double a, char *ex, char *ac) {
	trans te, ta;
	te.asDouble = e;
	ta.asDouble = a;
	if (te.asInt64 != ta.asInt64) {
		printf("[ Succeeded ]  (%s != %s) expect:%f [0x%016llx], actual:%f [0x%016llx]\n", ex, ac, e, e, a, a);
	} else {
		printf("[ Failed    ]  (%s != %s) expect:%f [0x%016llx], actual:%f [0x%016llx]\n", ex, ac, e, e, a, a);
	}
}

__device__
void Test::expect_ne(bool e, bool a, char *ex, char *ac) {
	if (e != a) {
		printf("[ Succeeded ]  (%s != %s) expect:%s, actual:%s\n", ex, ac, (e ? "True" : "False"), (a ? "True" : "False"));
	} else {
		printf("[ Failed    ]  (%s != %s) expect:%s, actual:%s\n", ex, ac, (e ? "True" : "False"), (a ? "True" : "False"));
	}
}



__device__
void Test::expect_eq2(double e1, double e2, double a1, double a2, char *ex1, char *ex2, char *ac1, char *ac2) {
	trans te1, te2, ta1, ta2;
	te1.asDouble = e1;
	te2.asDouble = e2;
	ta1.asDouble = a1;
	ta2.asDouble = a2;
	if (te1.asInt64 == ta1.asInt64 && te2.asInt64 == ta2.asInt64) {
		printf("[ Succeeded ]  ({%s, %s} == {%s, %s}) expect:{%lle, %lle} [{0x%016llx, 0x%016llx}], actual:{%lle, %lle} [{0x%016llx, 0x%016llx}]\n", ex1, ex2, ac1, ac2, e1, e2, e1, e2, a1, a2, a1, a2);
	} else {
		printf("[ Failed    ]  ({%s, %s} == {%s, %s}) expect:{%lle, %lle} [{0x%016llx, 0x%016llx}], actual:{%lle, %lle} [{0x%016llx, 0x%016llx}]\n", ex1, ex2, ac1, ac2, e1, e2, e1, e2, a1, a2, a1, a2);
	}
}

__device__
void Test::expect_ne2(double e1, double e2, double a1, double a2, char *ex1, char *ex2, char *ac1, char *ac2) {
	trans te1, te2, ta1, ta2;
	te1.asDouble = e1;
	te2.asDouble = e2;
	ta1.asDouble = a1;
	ta2.asDouble = a2;
	if (te1.asInt64 != ta1.asInt64 || te2.asInt64 != ta2.asInt64) {
		printf("[ Succeeded ]  ({%s, %s} != {%s, %s}) expect:{%lle, %lle} [{0x%016llx, 0x%016llx}], actual:{%lle, %lle} [{0x%016llx, 0x%016llx}]\n", ex1, ex2, ac1, ac2, e1, e2, e1, e2, a1, a2, a1, a2);
	} else {
		printf("[ Failed    ]  ({%s, %s} != {%s, %s}) expect:{%lle, %lle} [{0x%016llx, 0x%016llx}], actual:{%lle, %lle} [{0x%016llx, 0x%016llx}]\n", ex1, ex2, ac1, ac2, e1, e2, e1, e2, a1, a2, a1, a2);
	}
}
