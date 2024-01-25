package exer;

import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Order;
import org.junit.jupiter.api.Test;

import java.time.LocalDateTime;
import java.util.ArrayList;
import java.util.List;

import static org.junit.jupiter.api.Assertions.assertEquals;


public class MainTest {

    LocalDateTime t1, t2, t3, t4, t5, t6, t7, t8, t9, t10;
    Car c1, c2, c3;
    ParkingRecord p1, p2, p3, p4, p5, p6, p7, p8, p9;


    @BeforeEach
    public void TestData() {
        t1 = LocalDateTime.of(2023, 11, 5, 8, 10, 0);
        t2 = LocalDateTime.of(2023, 11, 5, 9, 30, 0);
        t3 = LocalDateTime.of(2023, 11, 5, 10, 30, 0);
        t4 = LocalDateTime.of(2023, 11, 5, 23, 10, 0);
        t5 = LocalDateTime.of(2023, 11, 7, 5, 10, 0);
        t6 = LocalDateTime.of(2023, 11, 7, 5, 20, 0);
        t7 = LocalDateTime.of(2023, 11, 9, 2, 0, 0);
        t8 = LocalDateTime.of(2023, 11, 9, 3, 50, 0);
        t9 = LocalDateTime.of(2023, 11, 9, 10, 0, 0);
        t10 = LocalDateTime.of(2023, 11, 11, 2, 30, 0);
        c1 = new Car("粤B11111");
        c2 = new Car("粤B11112");
        c3 = new Car("粤B11113");
        c1.setCalculate(new ExternalCalculate());
        c2.setCalculate(new VIPCalculate());
        c3.setCalculate(new InternalCalculate());
        p1 = new ParkingRecord(new TimeRange(t1, t2), c1);
        p2 = new ParkingRecord(new TimeRange(t5, t6), c1);
        p3 = new ParkingRecord(new TimeRange(t7, t10), c1);
        p4 = new ParkingRecord(new TimeRange(t2, t3), c2);
        p5 = new ParkingRecord(new TimeRange(t5, t6), c2);
        p6 = new ParkingRecord(new TimeRange(t8, t10), c2);
        p7 = new ParkingRecord(new TimeRange(t3, t4), c3);
        p8 = new ParkingRecord(new TimeRange(t7, t8), c3);
        p9 = new ParkingRecord(new TimeRange(t9, t10), c3);
    }


    @Test
    @Order(1)
    public void Test01ParkingHours() {
        assertEquals("1.3", String.format("%.1f", p1.getParkingHours()));
        assertEquals("0.2", String.format("%.1f", p2.getParkingHours()));
        assertEquals("48.5", String.format("%.1f", p3.getParkingHours()));
        assertEquals("1.0", String.format("%.1f", p4.getParkingHours()));
        assertEquals("0.2", String.format("%.1f", p5.getParkingHours()));
        assertEquals("46.7", String.format("%.1f", p6.getParkingHours()));
        assertEquals("12.7", String.format("%.1f", p7.getParkingHours()));
        assertEquals("1.8", String.format("%.1f", p8.getParkingHours()));
        assertEquals("40.5", String.format("%.1f", p9.getParkingHours()));
    }


    @Test
    @Order(2)
    public void Test02CarOwner1() {
        CarOwner c = new CarOwner("owner1");
        c.addParkingRecord(p1);
        c.addParkingRecord(p2);
        c.addParkingRecord(p3);
        assertEquals("[owner1] Parting 3 times, cost 150 RMB, activePoints is 7",
                c.toString());
        List<String> results = new ArrayList<>();
        results.add("粤B11111 -> parking 1.3 hours");
        results.add("粤B11111 -> parking 0.2 hours");
        results.add("粤B11111 -> parking 48.5 hours");
        for (int i = 0; i < c.getParkingRecords().size(); i++) {
            assertEquals(results.get(i), c.getParkingRecords().get(i).toString());
        }
    }

    @Test
    @Order(3)
    public void Test03CarOwner2() {
        CarOwner c = new CarOwner("owner2");
        c.addParkingRecord(p4);
        c.addParkingRecord(p5);
        c.addParkingRecord(p6);
        assertEquals("[owner2] Parting 3 times, cost 215 RMB, activePoints is 10",
                c.toString());
        List<String> results = new ArrayList<>();
        results.add("粤B11112 -> parking 1.0 hours");
        results.add("粤B11112 -> parking 0.2 hours");
        results.add("粤B11112 -> parking 46.7 hours");
        for (int i = 0; i < c.getParkingRecords().size(); i++) {
            assertEquals(results.get(i), c.getParkingRecords().get(i).toString());
        }
    }

    @Test
    @Order(4)
    public void Test04CarOwner3() {
        CarOwner c = new CarOwner("owner3");
        c.addParkingRecord(p7);
        c.addParkingRecord(p8);
        c.addParkingRecord(p9);
        assertEquals("[owner3] Parting 3 times, cost 170 RMB, activePoints is 5",
                c.toString());
        List<String> results = new ArrayList<>();
        results.add("粤B11113 -> parking 12.7 hours");
        results.add("粤B11113 -> parking 1.8 hours");
        results.add("粤B11113 -> parking 40.5 hours");
        for (int i = 0; i < c.getParkingRecords().size(); i++) {
            assertEquals(results.get(i), c.getParkingRecords().get(i).toString());
        }
    }


}
