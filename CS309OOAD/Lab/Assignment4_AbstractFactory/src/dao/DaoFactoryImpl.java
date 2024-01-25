package dao;

import java.io.BufferedInputStream;
import java.io.FileInputStream;
import java.io.IOException;
import java.io.InputStream;
import java.util.Properties;

public class DaoFactoryImpl implements DaoFactory{
    private static DaoFactoryImpl instance = null;

    private static Properties prop = new Properties();
    private DaoFactoryImpl() {
        InputStream in = null;
        try {
            in = new BufferedInputStream(new FileInputStream("resource/data.properties"));
            prop.load(in);
        } catch (IOException e) {
            throw new RuntimeException(e);
        }
        System.out.println(prop.get("factory"));
        System.out.println(prop.get("staff"));
        System.out.println(prop.get("computer"));
    }

    public Object createObject(String classname) {
        Class clz = null;
        try {
            clz = Class.forName(classname);
            return clz.getDeclaredConstructor().newInstance();
        } catch (Exception e) {
            throw new RuntimeException(e);
        }
    }

    public static DaoFactoryImpl getInstance(){
        if(instance == null){
            instance = new DaoFactoryImpl();
        }
        return instance;
    }
    public ComputerDao createComputerDao(){
        ComputerDao computerDao = (ComputerDao) createObject((String) prop.get("computer"));
        return computerDao;
    }
    public StaffDao createStaffDao(){
        StaffDao staffDao = (StaffDao) createObject((String) prop.get("staff"));
        return staffDao;
    }
}
